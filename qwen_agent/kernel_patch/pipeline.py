from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar

import json5

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from qwen_agent.log import logger

from .config import KernelPatchConfig
from .git_tools import GitRepository, parse_patch_hunks, summarize_text
from .mcp_backend import KernelRepoMCPClient, run_agentic_json_stage, run_agentic_text_stage
from .models import CodeEditPlan, FixPlan, KernelPatchRunResult, KernelPatchSessionState, RootCauseReport, SolverAttempt
from .prompts import build_decoder_prompt, build_planner_agentic_prompt, build_solver_agentic_prompt
from .trace import TraceRecorder

T = TypeVar('T')


class KernelPatchPipeline:

    def __init__(self,
                 llm=None,
                 debug_confirm: Optional[Callable[[KernelPatchConfig, FixPlan, int], bool]] = None):
        self.llm = llm
        self.debug_confirm = debug_confirm

    def run(self, config: KernelPatchConfig) -> KernelPatchRunResult:
        config.ensure_artifact_dir()
        trace = TraceRecorder(config.debug_log_path)
        origin_repo = GitRepository(config.origin_repo, trace=trace, label='origin')
        target_base_repo = GitRepository(config.target_repo, trace=trace, label='target_base')
        mcp_client = KernelRepoMCPClient(trace=trace) if config.code_query_via_mcp else None

        target_commit = config.target_commit or self._resolve_target_commit(config, target_base_repo, origin_repo, trace,
                                                                            mcp_client)
        target_repo_path = Path(config.target_repo).expanduser().resolve()
        target_repo = GitRepository(str(target_repo_path), trace=trace, label='target_checkout')
        original_head = target_repo.current_head()
        if not target_repo.is_clean():
            raise RuntimeError(
                f'Target repository {target_repo.path} is not clean; refusing direct checkout mode. '
                'Clean or clone the repo before running kernel_patch.')
        target_repo.checkout_detached(target_commit)
        state: Optional[KernelPatchSessionState] = None
        attempts: List[SolverAttempt] = []
        final_plan: Optional[FixPlan] = None

        try:
            origin_patch_payload = self._load_origin_patch_payload(config, origin_repo, mcp_client)
            state = KernelPatchSessionState(
                target_commit=target_commit,
                origin_patch=origin_patch_payload['patch_text'],
                commit_message=origin_patch_payload.get('commit_message', ''),
                changed_files=list(origin_patch_payload.get('changed_files', [])),
                origin_hunks=parse_patch_hunks(origin_patch_payload['patch_text']),
            )

            state.decoder_report = self._run_decoder_agent(state, trace)
            for iteration in range(1, config.max_iterations + 1):
                state.iteration = iteration
                target_repo.reset_hard(target_commit)

                final_plan = self._run_planner_agent(config, state, trace, str(target_repo_path))
                state.fix_plan = final_plan
                self._debug_pause(config, final_plan, iteration, trace)

                solver_output = self._run_solver_agent(config, state, trace, str(target_repo_path))
                state.solver_output = solver_output

                patch_text, apply_ok, apply_stderr, diff_ok, diff_check_stderr = self._materialize_patch(
                    solver_output=solver_output,
                    target_repo=target_repo,
                    patch_path=str(config.patch_path),
                    changed_files=state.changed_files,
                    target_commit=target_commit,
                )
                config.patch_path.write_text(patch_text, encoding='utf-8')
                state.apply_ok = apply_ok
                state.apply_stderr = apply_stderr
                state.diff_ok = diff_ok
                state.diff_check_stderr = diff_check_stderr

                attempts.append(
                    SolverAttempt(
                        iteration=iteration,
                        apply_check_passed=apply_ok and diff_ok,
                        patch_path=str(config.patch_path),
                        apply_check_stderr=apply_stderr or diff_check_stderr,
                        solver_raw_output=solver_output,
                        verification_summary='git apply --check passed' if apply_ok and diff_ok else
                        (apply_stderr or diff_check_stderr),
                        next_action='stop' if apply_ok and diff_ok else 'retry_solver',
                    ))
                trace.record_event(
                    phase='solver',
                    event_type='apply_check',
                    payload={
                        'iteration': iteration,
                        'apply_ok': apply_ok,
                        'diff_ok': diff_ok,
                        'apply_stderr': apply_stderr,
                        'diff_check_stderr': diff_check_stderr,
                    },
                )
                if apply_ok and diff_ok:
                    break
        finally:
            target_repo.reset_hard(target_commit)
            target_repo.checkout_detached(original_head)

        if state is None or final_plan is None or state.decoder_report is None:
            raise RuntimeError('KernelPatchPipeline did not produce decoder/planner outputs')

        success = bool(attempts and attempts[-1].apply_check_passed)
        self._write_analysis_markdown(config, state, attempts, success)
        trace_payload = trace.build_trace_payload(
            task=config.model_dump(),
            extra={
                'target_repo_path': str(target_repo_path),
                'session_state': state.model_dump(),
                'attempts': [attempt.model_dump() for attempt in attempts],
                'success': success,
            },
        )
        config.trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = 'git apply --check passed' if success else 'git apply --check failed after max_iterations'
        return KernelPatchRunResult(
            cve_id=config.cve_id,
            success=success,
            target_commit=target_commit,
            artifact_dir=str(config.artifact_dir),
            target_repo_path=str(target_repo_path),
            analysis_path=str(config.analysis_path),
            patch_path=str(config.patch_path),
            trace_path=str(config.trace_path),
            decoder_report=state.decoder_report,
            final_plan=final_plan,
            attempts=attempts,
            retrieval_report=None,
            verification_reports=[],
            repair_history=[],
            summary=summary,
        )

    def _load_origin_patch_payload(self, config: KernelPatchConfig, origin_repo: GitRepository,
                                   mcp_client: Optional[KernelRepoMCPClient]) -> Dict[str, Any]:
        if mcp_client is not None:
            return mcp_client.show_commit_patch(config.origin_repo, config.community_commit_id)
        patch_text = origin_repo.show_patch(config.community_commit_id)
        return {
            'commit_id': config.community_commit_id,
            'commit_message': origin_repo.commit_message(config.community_commit_id),
            'changed_files': origin_repo.changed_files(config.community_commit_id),
            'patch_text': patch_text,
        }

    def _run_decoder_agent(self, state: KernelPatchSessionState, trace: TraceRecorder) -> RootCauseReport:
        prompt = build_decoder_prompt(state.commit_message, state.changed_files, state.origin_patch, state.origin_hunks)
        return self._generate_structured('decoder', prompt, RootCauseReport, trace)

    def _run_planner_agent(self, config: KernelPatchConfig, state: KernelPatchSessionState, trace: TraceRecorder,
                           repo_path: str) -> FixPlan:
        if state.decoder_report is None:
            raise RuntimeError('Planner requires decoder_report')
        if config.code_query_via_mcp:
            prompt = build_planner_agentic_prompt(
                decoder_report=state.decoder_report,
                origin_hunks=state.origin_hunks,
                changed_files=state.changed_files,
                previous_apply_error=state.apply_stderr or state.diff_check_stderr,
                max_tool_calls=max(1, config.planner_max_tool_calls or config.retriever_max_tool_calls),
            )
            payload = run_agentic_json_stage(
                llm=self.llm,
                trace=trace,
                repo=repo_path,
                prompt=prompt,
                system_message=(
                    '你是 Linux 内核补丁 Planner Agent。必须基于目标仓真实代码做规划。'
                    '你可以使用仓库检索、读文件、git 和只读命令工具；不要臆测。'
                ),
                allowed_tool_suffixes=['search_code', 'read_range', 'read_file', 'list_files', 'run_git', 'run_command'],
                max_tool_calls=max(10, config.planner_max_tool_calls or config.retriever_max_tool_calls),
                timeout_sec=max(600, config.agentic_retrieval_timeout_sec),
            )
            plan = FixPlan.model_validate(payload)
        else:
            plan = FixPlan(
                summary='Non-MCP mode planner fallback',
                candidate_files=state.changed_files,
                planned_hunks=[],
                adaptation_notes=[],
                unresolved_risks=['code_query_via_mcp is disabled; planner did not inspect target repository'],
            )
        trace.add_step(step=f'planner.iteration_{state.iteration}', prompt='agentic_planner', output=plan.model_dump())
        return plan

    def _run_solver_agent(self, config: KernelPatchConfig, state: KernelPatchSessionState, trace: TraceRecorder,
                          repo_path: str) -> str:
        if state.decoder_report is None or state.fix_plan is None:
            raise RuntimeError('Solver requires decoder_report and fix_plan')
        prompt = build_solver_agentic_prompt(
            decoder_report=state.decoder_report,
            fix_plan=state.fix_plan,
            origin_patch=state.origin_patch,
            changed_files=state.changed_files,
            previous_apply_error=state.apply_stderr or state.diff_check_stderr,
            max_tool_calls=config.solver_max_tool_calls,
        )
        if config.code_query_via_mcp:
            output = run_agentic_text_stage(
                llm=self.llm,
                trace=trace,
                repo=repo_path,
                prompt=prompt,
                system_message=(
                    '你是 Linux 内核补丁 Solver Agent。你可以直接读写目标仓文件，并使用 git 或 shell 命令自检。'
                    '最终必须输出一个 JSON 对象。'
                ),
                allowed_tool_suffixes=[
                    'search_code', 'read_range', 'read_file', 'write_file', 'replace_in_file', 'replace_lines',
                    'replace_near_anchor', 'insert_before', 'insert_after', 'list_files', 'run_git', 'run_command'
                ],
                max_tool_calls=max(12, config.solver_max_tool_calls),
                timeout_sec=max(600, config.agentic_retrieval_timeout_sec),
            )
        else:
            output = json.dumps({'status': 'failed', 'summary': 'code_query_via_mcp is disabled', 'modified_files': []},
                                ensure_ascii=False)
        trace.add_step(step=f'solver.iteration_{state.iteration}', prompt=prompt, output=output)
        return output

    def _materialize_patch(self,
                           solver_output: str,
                           target_repo: GitRepository,
                           patch_path: str,
                           changed_files: List[str],
                           target_commit: str) -> tuple[str, bool, str, bool, str]:
        payload = None
        try:
            payload = self._extract_json(solver_output)
            if isinstance(payload, dict) and 'edits' in payload:
                edit_plan = CodeEditPlan.model_validate(payload)
                target_repo.apply_code_edits(edit_plan.edits)
        except Exception:
            pass

        diff_ok, diff_stderr = target_repo.diff_check()
        patch_text = target_repo.diff(changed_files) or target_repo.diff()
        Path(patch_path).write_text(patch_text, encoding='utf-8')
        if not patch_text.strip():
            if isinstance(payload, dict) and payload.get('status') in ('no_changes_needed', 'already_fixed'):
                return patch_text, True, '', True, ''
            return patch_text, False, 'solver produced no file changes', diff_ok, diff_stderr
        if not diff_ok:
            return patch_text, False, diff_stderr, diff_ok, diff_stderr
        target_repo.reset_hard(target_commit)
        apply_ok, apply_stderr = target_repo.apply_check(patch_path)
        return patch_text, apply_ok, apply_stderr, diff_ok, diff_stderr

    def _resolve_target_commit(self,
                               config: KernelPatchConfig,
                               target_repo: GitRepository,
                               origin_repo: GitRepository,
                               trace: TraceRecorder,
                               mcp_client: Optional[KernelRepoMCPClient]) -> str:
        if mcp_client is not None:
            for repo in (target_repo, origin_repo):
                try:
                    commit = mcp_client.resolve_parent_commit(str(repo.path), config.community_commit_id)
                    trace.record_event('setup', 'resolved_target_commit',
                                       {'repo': str(repo.path), 'target_commit': commit, 'via': 'mcp'})
                    return commit
                except Exception as ex:
                    trace.record_event('setup', 'resolve_target_commit_failed',
                                       {'repo': str(repo.path), 'error': str(ex), 'via': 'mcp'})
        for repo in (target_repo, origin_repo):
            try:
                commit = repo.parent_commit(config.community_commit_id)
                trace.record_event('setup', 'resolved_target_commit',
                                   {'repo': str(repo.path), 'target_commit': commit, 'via': 'git'})
                return commit
            except Exception as ex:
                trace.record_event('setup', 'resolve_target_commit_failed',
                                   {'repo': str(repo.path), 'error': str(ex), 'via': 'git'})
        raise RuntimeError(f'Unable to resolve parent commit for {config.community_commit_id}')

    def _generate_structured(self, step: str, prompt: str, response_model: Type[T], trace: TraceRecorder) -> T:
        response_text = self._call_llm(prompt)
        trace.add_step(step=step, prompt=prompt, output=response_text)
        payload = self._extract_json(response_text)
        parsed = response_model.model_validate(payload)
        trace.record_event(step, 'parsed_structured_output',
                           {'summary': summarize_text(json.dumps(parsed.model_dump(), ensure_ascii=False))})
        return parsed

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError('KernelPatchPipeline requires an llm instance or llm config')
        llm = self.llm
        if isinstance(llm, dict):
            llm = get_chat_model(llm)
            self.llm = llm
        responses = llm.chat(
            messages=[
                Message(role=SYSTEM, content='你是 Linux 内核补丁系统中的阶段 Agent。输出必须严格遵循用户要求。'),
                Message(role=USER, content=prompt),
            ],
            stream=False,
        )
        text_parts = [message.content for message in responses if message.role == ASSISTANT and isinstance(message.content, str)]
        response_text = '\n'.join(part for part in text_parts if part)
        if not response_text:
            responses = llm.chat(
                messages=[
                    Message(role=SYSTEM, content='你是 Linux 内核补丁系统中的阶段 Agent。输出必须严格遵循用户要求。'),
                    Message(role=USER, content=prompt),
                ],
                stream=False,
            )
            text_parts = [message.content for message in responses if message.role == ASSISTANT and isinstance(message.content, str)]
            response_text = '\n'.join(part for part in text_parts if part)
        logger.info('KernelPatchPipeline LLM response: %s', summarize_text(response_text))
        return response_text

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        fence_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned, flags=re.S)
        if fence_match:
            cleaned = fence_match.group(1)
        return json5.loads(re.search(r'(\{.*\})', cleaned, flags=re.S).group(1) if '{' in cleaned else cleaned)

    def _debug_pause(self, config: KernelPatchConfig, plan: FixPlan, iteration: int, trace: TraceRecorder) -> None:
        if not config.debug_mode:
            return
        trace.record_event('debug', 'await_confirmation',
                           {'iteration': iteration, 'summary': plan.summary, 'candidate_files': plan.candidate_files})
        if self.debug_confirm is not None:
            should_continue = self.debug_confirm(config, plan, iteration)
            if not should_continue:
                raise RuntimeError('Debug confirmation callback aborted the run')
            return
        answer = input(f'[Debug Mode] {config.cve_id} iteration {iteration} patch generation ready. Continue? [y/N]: ')
        if answer.strip().lower() not in ('y', 'yes'):
            raise RuntimeError('Debug confirmation aborted by user')

    def _write_analysis_markdown(self,
                                 config: KernelPatchConfig,
                                 state: KernelPatchSessionState,
                                 attempts: Iterable[SolverAttempt],
                                 success: bool) -> None:
        assert state.decoder_report is not None
        assert state.fix_plan is not None
        lines = [
            f'# {config.cve_id} Analysis',
            '',
            '## Task',
            f'- Community Commit: `{config.community_commit_id}`',
            f'- Target Commit: `{state.target_commit}`',
            '',
            '## Decoder',
            f'- Vulnerability Type: {state.decoder_report.vulnerability_type}',
            f'- Root Cause: {state.decoder_report.root_cause}',
            f'- Impacted Files: {", ".join(state.decoder_report.impacted_files) or "N/A"}',
            '',
            '## Planner',
            f'- Summary: {state.fix_plan.summary}',
            f'- Candidate Files: {", ".join(state.fix_plan.candidate_files) or "N/A"}',
            '',
            '## Solver Attempts',
        ]
        for attempt in attempts:
            status = 'PASS' if attempt.apply_check_passed else 'FAIL'
            lines.append(f'- Iteration {attempt.iteration}: {status}')
            if attempt.apply_check_stderr:
                lines.append(f'  stderr: {attempt.apply_check_stderr}')
        lines.extend(['', '## Result', f'- Success: {success}'])
        config.analysis_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
