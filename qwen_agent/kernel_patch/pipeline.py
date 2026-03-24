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
from .git_tools import GitRepository, build_target_snippets, parse_patch_hunks, prepare_worktree, summarize_text
from .mcp_backend import (KernelRepoMCPClient, build_target_snippets_via_mcp, run_agentic_json_stage,
                          run_agentic_text_stage)
from .models import CodeEditPlan, FixPlan, KernelPatchRunResult, RetrievalReport, RootCauseReport, SolverAttempt
from .prompts import (build_decoder_prompt, build_planner_agentic_prompt, build_planner_prompt,
                      build_solver_agentic_prompt, build_solver_prompt)
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
        if config.prepare_target_worktree:
            target_repo_path = prepare_worktree(target_base_repo, target_commit, config.worktree_path)
        target_repo = GitRepository(str(target_repo_path), trace=trace, label='target_worktree')
        verify_repo_path = prepare_worktree(target_base_repo, target_commit, config.verify_worktree_path)
        verify_repo = GitRepository(str(verify_repo_path), trace=trace, label='verify_worktree')

        if mcp_client is not None:
            origin_patch_payload = mcp_client.show_commit_patch(config.origin_repo, config.community_commit_id)
            origin_patch = origin_patch_payload['patch_text']
            origin_hunks = parse_patch_hunks(origin_patch)
            changed_files = list(origin_patch_payload.get('changed_files', []))
            commit_message = origin_patch_payload.get('commit_message', '')
        else:
            origin_patch = origin_repo.show_patch(config.community_commit_id)
            origin_hunks = parse_patch_hunks(origin_patch)
            changed_files = origin_repo.changed_files(config.community_commit_id)
            commit_message = origin_repo.commit_message(config.community_commit_id)

        decoder_prompt = build_decoder_prompt(commit_message, changed_files, origin_patch, origin_hunks)
        decoder_report = self._generate_structured(
            step='decoder',
            prompt=decoder_prompt,
            response_model=RootCauseReport,
            trace=trace,
        )

        attempts: List[SolverAttempt] = []
        previous_apply_error = ''
        final_plan: Optional[FixPlan] = None
        retrieval_reports: List[RetrievalReport] = []
        can_use_main_llm_agentic = config.code_query_via_mcp and (
            isinstance(self.llm, dict) or hasattr(self.llm, 'model'))

        for iteration in range(1, config.max_iterations + 1):
            retrieval_report: Optional[RetrievalReport] = None
            if not can_use_main_llm_agentic:
                if mcp_client is not None:
                    retrieval_report = build_target_snippets_via_mcp(
                        mcp_client=mcp_client,
                        repo=str(target_repo_path),
                        origin_hunks=origin_hunks,
                        semantic_anchors=decoder_report.semantic_anchors,
                        path_hints=changed_files,
                        symbol_hints=decoder_report.impacted_functions + decoder_report.impacted_macros +
                        decoder_report.impacted_structs + decoder_report.impacted_globals,
                        origin_patch_text=origin_patch,
                    )
                    retrieval_reports.append(retrieval_report)
                    target_snippets = retrieval_report.snippets
                else:
                    target_snippets = build_target_snippets(
                        target_repo=target_repo,
                        origin_hunks=origin_hunks,
                        semantic_anchors=decoder_report.semantic_anchors,
                        path_hints=changed_files,
                    )
                planner_prompt = build_planner_prompt(
                    decoder_report=decoder_report,
                    origin_hunks=origin_hunks,
                    target_snippets=target_snippets,
                    retrieval_report=retrieval_report,
                    previous_apply_error=previous_apply_error,
                )
                final_plan = self._generate_structured(
                    step=f'planner.iteration_{iteration}',
                    prompt=planner_prompt,
                    response_model=FixPlan,
                    trace=trace,
                )
            else:
                planner_prompt = build_planner_agentic_prompt(
                    decoder_report=decoder_report,
                    origin_hunks=origin_hunks,
                    changed_files=changed_files,
                    previous_apply_error=previous_apply_error,
                    max_tool_calls=config.agentic_retrieval_max_tool_calls,
                )
                planner_payload = run_agentic_json_stage(
                    llm=self.llm,
                    trace=trace,
                    repo=str(target_repo_path),
                    prompt=planner_prompt,
                    changed_files=changed_files,
                    system_message=(
                        '你是 Linux 内核补丁规划 Agent。必须通过 MCP 工具在 changed_files 内查代码，不允许臆测。'
                        '只允许使用 search_code 和 read_range，并在预算内输出 JSON。'
                    ),
                    max_tool_calls=config.agentic_retrieval_max_tool_calls,
                    timeout_sec=config.agentic_retrieval_timeout_sec,
                )
                final_plan = FixPlan.model_validate(planner_payload)
                target_snippets = []

            final_plan = self._attach_solver_snippets(
                fix_plan=final_plan,
                changed_files=changed_files,
                target_repo=target_repo,
                target_repo_path=str(target_repo_path),
                mcp_client=mcp_client,
                trace=trace,
            )

            self._debug_pause(config, final_plan, iteration, trace)

            solver_prompt = build_solver_prompt(
                decoder_report=decoder_report,
                fix_plan=final_plan,
                origin_patch=origin_patch,
                target_snippets=target_snippets,
                retrieval_report=retrieval_report,
                previous_apply_error=previous_apply_error,
            )
            solver_output = self._generate_text(
                step=f'solver.iteration_{iteration}',
                prompt=solver_prompt,
                trace=trace,
            )
            target_repo.reset_hard(target_commit)
            verify_repo.reset_hard(target_commit)
            patch_text, apply_ok, apply_stderr = self._materialize_patch_from_solver_output(
                solver_output=solver_output,
                target_repo=target_repo,
                verify_repo=verify_repo,
                patch_path=str(config.patch_path),
                target_commit=target_commit,
                changed_files=changed_files,
            )
            config.patch_path.write_text(patch_text, encoding='utf-8')
            if mcp_client is not None and not apply_ok:
                # keep stderr from verify_repo apply --check
                pass
            else:
                pass
            attempts.append(
                SolverAttempt(
                    iteration=iteration,
                    apply_check_passed=apply_ok,
                    patch_path=str(config.patch_path),
                    apply_check_stderr=apply_stderr,
                    solver_raw_output=solver_output,
                ))
            trace.record_event(
                phase='solver',
                event_type='apply_check',
                payload={
                    'iteration': iteration,
                    'patch_path': str(config.patch_path),
                    'success': apply_ok,
                    'stderr': apply_stderr,
                },
            )
            if apply_ok:
                break
            previous_apply_error = apply_stderr

        assert final_plan is not None
        success = bool(attempts and attempts[-1].apply_check_passed)
        self._write_analysis_markdown(config, decoder_report, final_plan, attempts, target_commit, success,
                                      retrieval_reports)
        trace_payload = trace.build_trace_payload(
            task=config.model_dump(),
            extra={
                'target_commit': target_commit,
                'decoder_report': decoder_report.model_dump(),
                'final_plan': final_plan.model_dump(),
                'attempts': [attempt.model_dump() for attempt in attempts],
                'success': success,
                'target_repo_path': str(target_repo_path),
                'retrieval_reports': [report.model_dump() for report in retrieval_reports],
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
            decoder_report=decoder_report,
            final_plan=final_plan,
            attempts=attempts,
            summary=summary,
        )

    def _materialize_patch_from_solver_output(self,
                                              solver_output: str,
                                              target_repo: GitRepository,
                                              verify_repo: GitRepository,
                                              patch_path: str,
                                              target_commit: str,
                                              changed_files: List[str]) -> tuple[str, bool, str]:
        try:
            payload = self._extract_json(solver_output)
            if isinstance(payload, dict) and 'edits' in payload:
                edit_plan = CodeEditPlan.model_validate(payload)
                target_repo.apply_code_edits(edit_plan.edits)
                diff_ok, diff_stderr = target_repo.diff_check()
                patch_text = target_repo.diff(changed_files)
                Path(patch_path).write_text(patch_text, encoding='utf-8')
                if not diff_ok:
                    return patch_text, False, diff_stderr
                apply_ok, apply_stderr = verify_repo.apply_check(patch_path)
                return patch_text, apply_ok, apply_stderr
        except Exception:
            pass

        patch_text = self._normalize_patch_text(solver_output)
        Path(patch_path).write_text(patch_text, encoding='utf-8')
        apply_ok, apply_stderr = verify_repo.apply_check(patch_path)
        return patch_text, apply_ok, apply_stderr

    def _attach_solver_snippets(self,
                                fix_plan: FixPlan,
                                changed_files: List[str],
                                target_repo: GitRepository,
                                target_repo_path: str,
                                mcp_client: Optional[KernelRepoMCPClient],
                                trace: TraceRecorder) -> FixPlan:
        if fix_plan.solver_snippets:
            return fix_plan

        snippets = []
        seen = set()

        def add_snippet(snippet):
            key = (snippet.file_path, snippet.start_line, snippet.end_line)
            if key in seen:
                return
            seen.add(key)
            snippets.append(snippet)

        for planned_hunk in fix_plan.planned_hunks:
            file_path = planned_hunk.file_path
            if file_path not in changed_files:
                continue
            matched = False
            if mcp_client is not None and planned_hunk.anchor.strip():
                try:
                    hits = mcp_client.search_code(target_repo_path, planned_hunk.anchor, path_glob=file_path)
                except Exception:
                    hits = []
                if hits:
                    hit = hits[0]
                    add_snippet(
                        mcp_client.read_range(
                            target_repo_path,
                            hit.file_path,
                            max(1, hit.line_number - 12),
                            hit.line_number + 28,
                            reason=f'solver_anchor:{planned_hunk.anchor}',
                        ))
                    matched = True
            if not matched and mcp_client is None and planned_hunk.anchor.strip():
                hits = target_repo.search_fixed_string(planned_hunk.anchor, path_glob=file_path)
                if hits:
                    hit = hits[0]
                    add_snippet(
                        target_repo.read_range(
                            hit.file_path,
                            max(1, hit.line_number - 12),
                            hit.line_number + 28,
                            reason=f'solver_anchor:{planned_hunk.anchor}',
                        ))
                    matched = True
            if not matched:
                if mcp_client is not None:
                    try:
                        add_snippet(mcp_client.read_range(target_repo_path, file_path, 1, 160,
                                                          reason=f'solver_file:{file_path}'))
                    except Exception:
                        pass
                else:
                    try:
                        add_snippet(target_repo.read_range(file_path, 1, 160, reason=f'solver_file:{file_path}'))
                    except Exception:
                        pass

        if not snippets:
            for file_path in changed_files:
                try:
                    if mcp_client is not None:
                        add_snippet(mcp_client.read_range(target_repo_path, file_path, 1, 160,
                                                          reason=f'solver_fallback:{file_path}'))
                    else:
                        add_snippet(target_repo.read_range(file_path, 1, 160, reason=f'solver_fallback:{file_path}'))
                except Exception:
                    continue
                if len(snippets) >= 2:
                    break

        trace.record_event(
            phase='solver',
            event_type='solver_snippets_attached',
            payload={
                'count': len(snippets),
                'files': [snippet.file_path for snippet in snippets],
            },
        )
        return fix_plan.model_copy(update={'solver_snippets': snippets})

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
                    trace.record_event(
                        phase='setup',
                        event_type='resolved_target_commit',
                        payload={'repo': str(repo.path), 'target_commit': commit, 'via': 'mcp'},
                    )
                    return commit
                except Exception as ex:  # noqa
                    trace.record_event(
                        phase='setup',
                        event_type='resolve_target_commit_failed',
                        payload={'repo': str(repo.path), 'error': str(ex), 'via': 'mcp'},
                    )
        for repo in (target_repo, origin_repo):
            try:
                commit = repo.parent_commit(config.community_commit_id)
                trace.record_event(
                    phase='setup',
                    event_type='resolved_target_commit',
                    payload={'repo': str(repo.path), 'target_commit': commit, 'via': 'git'},
                )
                return commit
            except Exception as ex:  # noqa
                trace.record_event(
                    phase='setup',
                    event_type='resolve_target_commit_failed',
                    payload={'repo': str(repo.path), 'error': str(ex), 'via': 'git'},
                )
        raise RuntimeError(f'Unable to resolve parent commit for {config.community_commit_id}')

    def _generate_structured(self, step: str, prompt: str, response_model: Type[T], trace: TraceRecorder) -> T:
        response_text = self._call_llm(prompt)
        trace.add_step(step=step, prompt=prompt, output=response_text)
        payload = self._extract_json(response_text)
        parsed = response_model.model_validate(payload)
        trace.record_event(
            phase=step,
            event_type='parsed_structured_output',
            payload={'summary': summarize_text(json.dumps(parsed.model_dump(), ensure_ascii=False))},
        )
        return parsed

    def _generate_text(self, step: str, prompt: str, trace: TraceRecorder) -> str:
        response_text = self._call_llm(prompt)
        trace.add_step(step=step, prompt=prompt, output=response_text)
        return response_text

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError('KernelPatchPipeline requires an llm instance or llm config')
        llm = self.llm
        if isinstance(llm, dict):
            llm = get_chat_model(llm)
            self.llm = llm
        responses = llm.chat(
            messages=[
                Message(role=SYSTEM, content='你是 Linux 内核 CVE 自动化补丁修复 Agent。输出必须严格遵循用户要求。'),
                Message(role=USER, content=prompt),
            ],
            stream=False,
        )
        text_parts = []
        for message in responses:
            if message.role == ASSISTANT and isinstance(message.content, str):
                text_parts.append(message.content)
        response_text = '\n'.join(part for part in text_parts if part)
        logger.info('KernelPatchPipeline LLM response: %s', summarize_text(response_text))
        return response_text

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        fence_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned, flags=re.S)
        if fence_match:
            cleaned = fence_match.group(1)
        try:
            return json5.loads(cleaned)
        except Exception:
            pass
        object_match = re.search(r'(\{.*\})', cleaned, flags=re.S)
        if object_match:
            return json5.loads(object_match.group(1))
        raise ValueError(f'Unable to parse JSON from LLM output: {cleaned}')

    def _normalize_patch_text(self, patch_text: str) -> str:
        cleaned = patch_text.strip()
        fence_match = re.search(r'```(?:diff|patch)?\s*(.*?)\s*```', cleaned, flags=re.S)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        if not cleaned.endswith('\n'):
            cleaned += '\n'
        return cleaned

    def _debug_pause(self, config: KernelPatchConfig, plan: FixPlan, iteration: int, trace: TraceRecorder) -> None:
        if not config.debug_mode:
            return
        trace.record_event(
            phase='debug',
            event_type='await_confirmation',
            payload={
                'iteration': iteration,
                'summary': plan.summary,
                'candidate_files': plan.candidate_files,
            },
        )
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
                                 decoder_report: RootCauseReport,
                                 fix_plan: FixPlan,
                                 attempts: Iterable[SolverAttempt],
                                 target_commit: str,
                                 success: bool,
                                 retrieval_reports: Iterable[RetrievalReport]) -> None:
        lines = [
            f'# {config.cve_id} Analysis',
            '',
            '## Task',
            f'- Community Commit: `{config.community_commit_id}`',
            f'- Target Commit: `{target_commit}`',
            f'- Origin Repo: `{config.origin_repo}`',
            f'- Target Repo: `{config.target_repo}`',
            '',
            '## Decoder',
            f'- Vulnerability Type: {decoder_report.vulnerability_type}',
            f'- Root Cause: {decoder_report.root_cause}',
            f'- Impacted Files: {", ".join(decoder_report.impacted_files) or "N/A"}',
            f'- Impacted Functions: {", ".join(decoder_report.impacted_functions) or "N/A"}',
            f'- Impacted Macros: {", ".join(decoder_report.impacted_macros) or "N/A"}',
            f'- Impacted Structs: {", ".join(decoder_report.impacted_structs) or "N/A"}',
            f'- Impacted Globals: {", ".join(decoder_report.impacted_globals) or "N/A"}',
            '',
            '### Fix Logic',
        ]
        lines.extend([f'- {item}' for item in decoder_report.fix_logic] or ['- N/A'])
        lines.extend(['', '### Semantic Anchors'])
        lines.extend([f'- {item}' for item in decoder_report.semantic_anchors] or ['- N/A'])
        retrieval_reports = list(retrieval_reports)
        if retrieval_reports:
            latest_retrieval = retrieval_reports[-1]
            covered_hunks = sum(1 for item in latest_retrieval.hunk_coverages if item.covered)
            lines.extend([
                '',
                '## Retriever',
                f'- Covered Hunks: {covered_hunks}/{len(latest_retrieval.hunk_coverages)}',
                f'- Retrieval Snippets: {len(latest_retrieval.snippets)}',
                f'- Missing Entities: {", ".join(latest_retrieval.missing_entities) or "N/A"}',
                f'- Added Functions: {", ".join(latest_retrieval.added_functions) or "N/A"}',
                f'- Added Macros: {", ".join(latest_retrieval.added_macros) or "N/A"}',
                f'- Added Structs: {", ".join(latest_retrieval.added_structs) or "N/A"}',
                f'- Added Includes: {", ".join(latest_retrieval.added_includes) or "N/A"}',
                '',
                '### Hunk Coverage',
            ])
            if latest_retrieval.hunk_coverages:
                for coverage in latest_retrieval.hunk_coverages:
                    status = 'COVERED' if coverage.covered else 'MISSING'
                    lines.append(f'- Hunk {coverage.hunk_index} `{coverage.file_path}`: {status}')
                    if coverage.missing_anchors:
                        lines.append(f'  missing anchors: {", ".join(coverage.missing_anchors)}')
                    if coverage.evidence:
                        evidence = coverage.evidence[0]
                        lines.append(
                            f'  evidence: {evidence.strategy} {evidence.file_path}:{evidence.start_line}-{evidence.end_line}'
                        )
            else:
                lines.append('- N/A')
            lines.extend(['', '### Entity Coverage'])
            if latest_retrieval.entity_coverages:
                for coverage in latest_retrieval.entity_coverages:
                    status = 'FOUND' if coverage.found else 'MISSING'
                    lines.append(f'- {coverage.entity_type} `{coverage.entity_name}`: {status}')
            else:
                lines.append('- N/A')
        lines.extend(['', '## Planner', f'- Summary: {fix_plan.summary}', '', '### Candidate Files'])
        lines.extend([f'- {item}' for item in fix_plan.candidate_files] or ['- N/A'])
        lines.extend(['', '### Planned Hunks'])
        if fix_plan.planned_hunks:
            for hunk in fix_plan.planned_hunks:
                lines.append(f'- `{hunk.file_path}` [{hunk.change_type}] {hunk.rationale}')
                lines.append(f'  anchor: {hunk.anchor}')
                lines.append(f'  adaptation: {hunk.adaptation_strategy}')
        else:
            lines.append('- N/A')
        lines.extend(['', '### Adaptation Notes'])
        lines.extend([f'- {item}' for item in fix_plan.adaptation_notes] or ['- N/A'])
        lines.extend(['', '### Unresolved Risks'])
        lines.extend([f'- {item}' for item in fix_plan.unresolved_risks] or ['- N/A'])
        lines.extend(['', '## Solver'])
        for attempt in attempts:
            status = 'PASS' if attempt.apply_check_passed else 'FAIL'
            lines.append(f'- Iteration {attempt.iteration}: {status}')
            if attempt.apply_check_stderr:
                lines.append(f'  stderr: {attempt.apply_check_stderr}')
        lines.extend(['', '## Result', f'- Success: {success}'])
        config.analysis_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
