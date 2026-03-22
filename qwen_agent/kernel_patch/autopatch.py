"""面向漏洞输入的 closed-loop autopatch 流水线。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import Message

from .autopatch_models import (AutoPatchResult, CodeWindow, PatchContext, PatchEdit, PatchPlan, PatchValidationOutcome,
                               RetrievalCandidate, SearchHit, VulnerabilityInput)
from .compile_validator import KernelCompileValidator
from .git_ops import KernelRepoManager
from .models import PatchCandidate
from .repo_access import LocalRepoMCPClient

JSON_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)
IDENT_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b')


def _extract_json_blob(text: str) -> Optional[dict]:
    match = JSON_FENCE_RE.search(text)
    candidates = [match.group(1)] if match else []
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        candidates.append(stripped)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    start = text.find('{')
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    chunk = text[start:index + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + '\n...\n[TRUNCATED]\n...\n' + text[-tail:]


def _render_patch_context(context: PatchContext, max_chars: int = 24000) -> str:
    sections = []
    if context.summary:
        sections.append('## Retrieval Summary\n' + context.summary)
    for window in context.windows:
        sections.append(f'## {window.path}:{window.start_line}-{window.end_line} [{window.reason}]\n{window.content}')
    return _truncate('\n\n'.join(sections).strip(), max_chars)


class PatchReasoner(Protocol):
    """推理器接口。"""

    def build_plan(self, vulnerability: VulnerabilityInput, context: PatchContext) -> PatchPlan:
        ...

    def synthesize_patch(self,
                         vulnerability: VulnerabilityInput,
                         context: PatchContext,
                         plan: PatchPlan,
                         feedback_text: str = '') -> PatchCandidate:
        ...


class HeuristicCandidateRetriever:
    """多路证据召回候选代码位置。"""

    def __init__(self, repo_client: LocalRepoMCPClient, allow_repo_fallback: bool = False):
        self.repo_client = repo_client
        self.allow_repo_fallback = allow_repo_fallback

    def _path_bonus(self, vulnerability: VulnerabilityInput, path: str) -> float:
        bonus = 0.0
        for hint in vulnerability.file_hints:
            normalized = hint.strip()
            if not normalized:
                continue
            if path == normalized:
                bonus += 8.0
            elif path.endswith(normalized) or normalized in path:
                bonus += 4.0
        for subsystem in vulnerability.subsystem_hints:
            if subsystem and f'/{subsystem.strip("/")}/' in f'/{path}':
                bonus += 2.5
        return bonus

    def _symbol_bonus(self, vulnerability: VulnerabilityInput, preview: str) -> float:
        bonus = 0.0
        for symbol in vulnerability.symbol_hints:
            if symbol and symbol in preview:
                bonus += 3.0
        return bonus

    def _query_terms(self, vulnerability: VulnerabilityInput) -> List[str]:
        terms = vulnerability.candidate_terms(limit=20)
        prioritized = []
        for symbol in vulnerability.symbol_hints:
            if symbol and symbol not in prioritized:
                prioritized.append(symbol)
        for term in terms:
            if term not in prioritized:
                prioritized.append(term)
        return prioritized[:20]

    def _scoped_paths(self, vulnerability: VulnerabilityInput, worktree_path: Path) -> Optional[List[str]]:
        scoped: List[str] = []
        seen = set()

        related_paths = vulnerability.metadata.get('related_paths', []) if vulnerability.metadata else []
        for raw_path in list(vulnerability.file_hints) + list(related_paths):
            normalized = str(raw_path).strip()
            if not normalized or normalized in seen:
                continue
            if (worktree_path / normalized).exists():
                scoped.append(normalized)
                seen.add(normalized)

        if scoped:
            return scoped

        for raw_path in vulnerability.subsystem_hints:
            normalized = str(raw_path).strip().strip('/')
            if not normalized or normalized in seen:
                continue
            if (worktree_path / normalized).exists():
                scoped.append(normalized)
                seen.add(normalized)

        return scoped or None

    def retrieve(self,
                 vulnerability: VulnerabilityInput,
                 worktree_path: Path,
                 max_candidates: int = 12) -> List[RetrievalCandidate]:
        aggregated: Dict[tuple[str, int], RetrievalCandidate] = {}
        scoped_paths = self._scoped_paths(vulnerability, worktree_path)

        for rank, term in enumerate(self._query_terms(vulnerability), start=1):
            hits = self.repo_client.search_literal(worktree_path, term, paths=scoped_paths, max_results=12)
            if not hits and not scoped_paths and self.allow_repo_fallback:
                hits = self.repo_client.search_literal(worktree_path, term, paths=None, max_results=12)
            for hit in hits:
                key = (hit.path, hit.line_no)
                base = max(0.5, 4.0 - rank * 0.15)
                score = base + self._path_bonus(vulnerability, hit.path) + self._symbol_bonus(vulnerability, hit.preview)
                reason = f'matched term `{term}` at {hit.path}:{hit.line_no}'
                if key not in aggregated:
                    aggregated[key] = RetrievalCandidate(path=hit.path,
                                                         line_no=hit.line_no,
                                                         score=score,
                                                         reason=reason,
                                                         symbol=term if term in vulnerability.symbol_hints else '',
                                                         preview=hit.preview,
                                                         supporting_hits=[SearchHit(path=hit.path,
                                                                                    line_no=hit.line_no,
                                                                                    preview=hit.preview,
                                                                                    score=score,
                                                                                    term=term)])
                else:
                    candidate = aggregated[key]
                    candidate.score += score * 0.5
                    candidate.supporting_hits.append(
                        SearchHit(path=hit.path, line_no=hit.line_no, preview=hit.preview, score=score, term=term))
                    if not candidate.symbol and term in vulnerability.symbol_hints:
                        candidate.symbol = term

        for hinted_path in (scoped_paths or vulnerability.file_hints):
            if hinted_path and (worktree_path / hinted_path).exists():
                key = (hinted_path, 1)
                if key not in aggregated:
                    aggregated[key] = RetrievalCandidate(path=hinted_path,
                                                         line_no=1,
                                                         score=2.0,
                                                         reason=f'file hint `{hinted_path}`',
                                                         preview='file_hint')

        candidates = list(aggregated.values())
        candidates.sort(key=lambda item: (-item.score, item.path, item.line_no))
        return candidates[:max_candidates]


class LocalContextAssembler:
    """将候选点扩展为局部上下文窗口。"""

    def __init__(self, repo_client: LocalRepoMCPClient):
        self.repo_client = repo_client

    def assemble(self,
                 vulnerability: VulnerabilityInput,
                 worktree_path: Path,
                 candidates: Sequence[RetrievalCandidate],
                 max_windows: int = 10) -> PatchContext:
        windows: List[CodeWindow] = []
        seen = set()
        for candidate in candidates[:max_windows]:
            window = self.repo_client.surrounding_window(worktree_path,
                                                        candidate.path,
                                                        candidate.line_no,
                                                        radius=28,
                                                        reason=candidate.reason)
            key = (window.path, window.start_line, window.end_line)
            if key not in seen:
                seen.add(key)
                windows.append(window)

            if candidate.path in vulnerability.file_hints:
                head = self.repo_client.read_file_head(worktree_path,
                                                      candidate.path,
                                                      max_lines=80,
                                                      reason=f'file head for {candidate.path}')
                key = (head.path, head.start_line, head.end_line)
                if key not in seen:
                    seen.add(key)
                    windows.append(head)

        summary_lines = [f'- {candidate.path}:{candidate.line_no} score={candidate.score:.2f} {candidate.reason}'
                         for candidate in candidates[:max_windows]]
        return PatchContext(candidates=list(candidates), windows=windows, summary='\n'.join(summary_lines))

    def render(self, context: PatchContext, max_chars: int = 24000) -> str:
        return _render_patch_context(context, max_chars=max_chars)


class LLMJsonPatchReasoner:
    """使用 LLM 输出结构化 plan 和最终 patch。"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _chat_text(self, prompt: str) -> str:
        response = self.llm.chat(messages=[Message(role='user', content=prompt)], stream=False)
        parts: List[str] = []
        for item in response:
            content = item.get('content') if isinstance(item, dict) else item.content
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        return '\n'.join(parts).strip()

    def _prompt_plan(self, vulnerability: VulnerabilityInput, context: PatchContext) -> str:
        rendered_context = _render_patch_context(context)
        return f"""You are a Linux kernel patch planning agent.

Your task is to analyze the vulnerability input and produce a JSON patch plan.
Use only the retrieved local repository context. Do not fabricate unseen files.

Return JSON only with this schema:
{{
  "root_cause": "short explanation",
  "invariant": "what must become true after the fix",
  "affected_paths": ["path1", "path2"],
  "retrieval_notes": ["note1", "note2"],
  "compile_targets": ["optional make targets or paths"],
  "confidence": 0.0,
  "edits": [
    {{
      "path": "repo/relative/path.c",
      "operation": "modify|add|delete",
      "intent": "what the edit does",
      "reason": "why this exact location is relevant",
      "target_snippet": "existing code to modify",
      "before_snippet": "anchor before insertion",
      "after_snippet": "anchor after insertion",
      "insert_snippet": "new code to insert",
      "delete_snippet": "code expected to be removed",
      "confidence": 0.0
    }}
  ]
}}

## Vulnerability Input
{vulnerability.narrative()}

## Local Repository Context
{rendered_context}
""".strip()

    def _prompt_patch(self,
                      vulnerability: VulnerabilityInput,
                      context: PatchContext,
                      plan: PatchPlan,
        feedback_text: str = '') -> str:
        rendered_context = _render_patch_context(context)
        plan_json = json.dumps(
            {
                'root_cause': plan.root_cause,
                'invariant': plan.invariant,
                'affected_paths': plan.affected_paths,
                'retrieval_notes': plan.retrieval_notes,
                'compile_targets': plan.compile_targets,
                'confidence': plan.confidence,
                'edits': [edit.__dict__ for edit in plan.edits],
            },
            ensure_ascii=False,
            indent=2,
        )
        prompt = f"""You are generating a Linux kernel unified diff patch.

Requirements:
- Output a unified diff patch only.
- Keep the patch minimal and focused.
- Respect the patch plan and the repository context.
- Prefer exact surrounding context from the retrieved windows.
- If adding code, place it using before/after anchors from the plan.
- If deleting code, ensure the deleted snippet matches existing local code.

## Vulnerability Input
{vulnerability.narrative()}

## Patch Plan
{plan_json}

## Local Repository Context
{rendered_context}
"""
        if feedback_text.strip():
            prompt += f'\n\n## Validation Feedback\n{feedback_text.strip()}'
        return prompt.strip()

    def _fallback_plan(self, vulnerability: VulnerabilityInput, context: PatchContext) -> PatchPlan:
        candidates = context.candidates[:3]
        affected_paths = [candidate.path for candidate in candidates]
        edits = []
        for candidate in candidates[:1]:
            edits.append(
                PatchEdit(path=candidate.path,
                          operation='modify',
                          intent='inspect and patch the most suspicious local code path',
                          reason=candidate.reason,
                          confidence=min(0.4 + candidate.score / 20.0, 0.75)))
        return PatchPlan(root_cause=vulnerability.title or 'Root cause requires model analysis.',
                         invariant='The dangerous path must be guarded and the fix must apply cleanly.',
                         affected_paths=affected_paths,
                         retrieval_notes=[candidate.reason for candidate in candidates],
                         edits=edits,
                         confidence=0.35)

    def build_plan(self, vulnerability: VulnerabilityInput, context: PatchContext) -> PatchPlan:
        response = self._chat_text(self._prompt_plan(vulnerability, context))
        payload = _extract_json_blob(response)
        if not payload:
            return self._fallback_plan(vulnerability, context)

        edits = []
        for raw_edit in payload.get('edits', []):
            if not isinstance(raw_edit, dict) or not raw_edit.get('path'):
                continue
            edits.append(
                PatchEdit(path=str(raw_edit.get('path', '')).strip(),
                          operation=str(raw_edit.get('operation', 'modify')).strip() or 'modify',
                          intent=str(raw_edit.get('intent', '')).strip(),
                          reason=str(raw_edit.get('reason', '')).strip(),
                          target_snippet=str(raw_edit.get('target_snippet', '')).rstrip(),
                          before_snippet=str(raw_edit.get('before_snippet', '')).rstrip(),
                          after_snippet=str(raw_edit.get('after_snippet', '')).rstrip(),
                          insert_snippet=str(raw_edit.get('insert_snippet', '')).rstrip(),
                          delete_snippet=str(raw_edit.get('delete_snippet', '')).rstrip(),
                          confidence=float(raw_edit.get('confidence', 0.0) or 0.0)))

        if not edits:
            return self._fallback_plan(vulnerability, context)

        return PatchPlan(root_cause=str(payload.get('root_cause', '')).strip(),
                         invariant=str(payload.get('invariant', '')).strip(),
                         affected_paths=[str(item).strip() for item in payload.get('affected_paths', []) if str(item).strip()],
                         retrieval_notes=[str(item).strip() for item in payload.get('retrieval_notes', []) if str(item).strip()],
                         edits=edits,
                         compile_targets=[str(item).strip() for item in payload.get('compile_targets', [])
                                          if str(item).strip()],
                         confidence=float(payload.get('confidence', 0.0) or 0.0))

    def synthesize_patch(self,
                         vulnerability: VulnerabilityInput,
                         context: PatchContext,
                         plan: PatchPlan,
                         feedback_text: str = '') -> PatchCandidate:
        response = self._chat_text(self._prompt_patch(vulnerability, context, plan, feedback_text=feedback_text))
        from .models import extract_patch_from_response, strip_patch_from_response

        patch_text = extract_patch_from_response(response)
        analysis = strip_patch_from_response(response, patch_text).strip() if patch_text else response.strip()
        return PatchCandidate(analysis_text=analysis, patch_text=patch_text, raw_response=response)


class ClosedLoopPatchValidator:
    """面向真实漏洞输入场景的 patch 验证器。"""

    def __init__(self,
                 manager: KernelRepoManager,
                 compile_validator: Optional[KernelCompileValidator] = None):
        self.manager = manager
        self.compile_validator = compile_validator

    def validate(self,
                 worktree_path: Path,
                 candidate: PatchCandidate,
                 plan: PatchPlan,
                 attempt_dir: Path) -> PatchValidationOutcome:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if not candidate.patch_found:
            return PatchValidationOutcome(patch_found=False,
                                          patch_apply_ok=False,
                                          diagnostics='Model did not return a valid unified diff patch.')

        patch_text = candidate.patch_text or ''
        self.manager.write_patch_file(attempt_dir, 'candidate.patch', patch_text)
        check = self.manager.check_patch(worktree_path, patch_text)
        if check.returncode != 0:
            diagnostics = (check.stdout + '\n' + check.stderr).strip()
            return PatchValidationOutcome(patch_found=True, patch_apply_ok=False, diagnostics=diagnostics)

        applied = self.manager.apply_patch(worktree_path, patch_text)
        if applied.returncode != 0:
            diagnostics = (applied.stdout + '\n' + applied.stderr).strip()
            return PatchValidationOutcome(patch_found=True, patch_apply_ok=False, diagnostics=diagnostics)

        changed_files = self.manager.current_changed_files(worktree_path)
        generated_patch = self.manager.export_current_patch(worktree_path)
        compile_validation = None
        diagnostics = 'Patch applied cleanly.'
        if self.compile_validator is not None:
            compile_targets = plan.compile_targets or changed_files
            compile_validation = self.compile_validator.validate(worktree_path, compile_targets)
            if compile_validation.status == 'failed':
                diagnostics = ('Patch applies cleanly but compile validation failed.\n'
                               f'Command: {compile_validation.command}\n'
                               f'Output:\n{compile_validation.output}')

        self.manager.write_patch_file(attempt_dir, 'applied.patch', generated_patch)
        return PatchValidationOutcome(patch_found=True,
                                      patch_apply_ok=True,
                                      changed_files=changed_files,
                                      diagnostics=diagnostics,
                                      generated_patch=generated_patch,
                                      compile_validation=compile_validation)


class ClosedLoopPatchPipeline:
    """从漏洞输入到 patch 生成与验证的闭环流水线。"""

    def __init__(self,
                 manager: KernelRepoManager,
                 repo_client: Optional[LocalRepoMCPClient] = None,
                 reasoner: Optional[PatchReasoner] = None,
                 compile_validator: Optional[KernelCompileValidator] = None):
        self.manager = manager
        self.repo_client = repo_client or LocalRepoMCPClient(manager)
        self.reasoner = reasoner
        self.retriever = HeuristicCandidateRetriever(self.repo_client)
        self.context_assembler = LocalContextAssembler(self.repo_client)
        self.validator = ClosedLoopPatchValidator(manager, compile_validator=compile_validator)

    def _artifact_dir(self, session_name: str) -> Path:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', session_name.strip()).strip('._') or 'autopatch'
        return self.manager.artifacts_root / safe

    def _augment_context_with_plan(self,
                                   worktree_path: Path,
                                   context: PatchContext,
                                   plan: PatchPlan) -> PatchContext:
        windows = list(context.windows)
        seen = {(window.path, window.start_line, window.end_line) for window in windows}
        notes = [context.summary] if context.summary else []
        notes.extend(plan.retrieval_notes)

        for edit in plan.edits:
            if edit.target_snippet.strip():
                matches = self.repo_client.locate_snippet(worktree_path, edit.path, edit.target_snippet, max_candidates=1)
                if matches:
                    match = matches[0]
                    window = self.repo_client.read_range(worktree_path,
                                                         edit.path,
                                                         max(1, match.start_line - 12),
                                                         match.end_line + 12,
                                                         reason=f'edit target match ({match.strategy})')
                    key = (window.path, window.start_line, window.end_line)
                    if key not in seen:
                        seen.add(key)
                        windows.append(window)
                    notes.append(
                        f'edit target matched at {edit.path}:{match.start_line}-{match.end_line} score={match.score:.2f}')
                    continue

            if edit.before_snippet.strip() or edit.after_snippet.strip():
                anchor = self.repo_client.resolve_insertion_point(worktree_path,
                                                                 edit.path,
                                                                 before_snippet=edit.before_snippet,
                                                                 after_snippet=edit.after_snippet)
                if anchor:
                    window = self.repo_client.surrounding_window(worktree_path,
                                                                 edit.path,
                                                                 anchor.line_no,
                                                                 radius=14,
                                                                 reason=f'insertion anchor ({anchor.strategy})')
                    key = (window.path, window.start_line, window.end_line)
                    if key not in seen:
                        seen.add(key)
                        windows.append(window)
                    notes.append(
                        f'insertion anchor resolved at {edit.path}:{anchor.line_no} strategy={anchor.strategy} score={anchor.score:.2f}'
                    )

        return PatchContext(candidates=context.candidates, windows=windows, summary='\n'.join(note for note in notes if note))

    def run(self,
            vulnerability: VulnerabilityInput,
            worktree_path: Path,
            session_name: str = 'autopatch',
            max_iterations: int = 2) -> AutoPatchResult:
        if self.reasoner is None:
            raise ValueError('ClosedLoopPatchPipeline requires a reasoner.')

        candidates = self.retriever.retrieve(vulnerability, worktree_path)
        context = self.context_assembler.assemble(vulnerability, worktree_path, candidates)
        plan = self.reasoner.build_plan(vulnerability, context)
        if plan.edits:
            context = self._augment_context_with_plan(worktree_path, context, plan)
        feedback = ''
        candidate = PatchCandidate(analysis_text='', patch_text=None, raw_response='')
        validation = PatchValidationOutcome(patch_found=False, patch_apply_ok=False, diagnostics='No attempt executed.')

        for attempt in range(1, max_iterations + 1):
            candidate = self.reasoner.synthesize_patch(vulnerability, context, plan, feedback_text=feedback)
            validation = self.validator.validate(worktree_path,
                                                 candidate,
                                                 plan,
                                                 self._artifact_dir(session_name) / f'attempt_{attempt:02d}')
            compile_failed = bool(validation.compile_validation and validation.compile_validation.status == 'failed')
            if validation.patch_apply_ok and not compile_failed:
                return AutoPatchResult(plan=plan,
                                       context=context,
                                       candidate=candidate,
                                       validation=validation,
                                       worktree_path=str(worktree_path),
                                       iterations=attempt)
            feedback = validation.diagnostics

        return AutoPatchResult(plan=plan,
                               context=context,
                               candidate=candidate,
                               validation=validation,
                               worktree_path=str(worktree_path),
                               iterations=max_iterations)

    def run_on_revision(self,
                        vulnerability: VulnerabilityInput,
                        revision: str = 'HEAD',
                        session_name: str = 'autopatch',
                        recreate_worktree: bool = True,
                        max_iterations: int = 2) -> AutoPatchResult:
        worktree_path = self.manager.create_detached_worktree(revision, session_name, recreate=recreate_worktree)
        return self.run(vulnerability, worktree_path, session_name=session_name, max_iterations=max_iterations)
