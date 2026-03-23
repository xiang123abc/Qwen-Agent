from __future__ import annotations

import json
import re
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import json5

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.tools import MCPManager

from .git_tools import summarize_text
from .models import (CodeSearchHit, CodeSnippet, EntityCoverageReport, HunkCoverageReport, PatchHunk,
                     RetrievalEvidence, RetrievalReport, RootCauseReport)
from .prompts import build_agentic_retriever_prompt
from .trace import TraceRecorder

ENTITY_BLACKLIST = {
    'if',
    'for',
    'while',
    'switch',
    'return',
    'sizeof',
    'READ_ONCE',
    'WRITE_ONCE',
    'list_for_each_entry',
    'list_entry',
}


class KernelRepoMCPClient:

    def __init__(self, trace: TraceRecorder, server_name: str = 'kernel_repo'):
        self.trace = trace
        self.server_name = server_name
        self._tool_map = self._init_tools()

    def _init_tools(self):
        config = build_kernel_repo_mcp_config(self.server_name)
        tools = MCPManager().initConfig(config)
        tool_map = {tool.name: tool for tool in tools}
        return tool_map

    def _call_tool(self, tool_suffix: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = f'{self.server_name}-{tool_suffix}'
        tool = self._tool_map[tool_name]
        self.trace.record_event(
            phase='tool',
            event_type='mcp_tool_start',
            payload={'tool_name': tool_name, 'arguments': arguments},
        )
        raw_result = tool.call(json.dumps(arguments, ensure_ascii=False))
        self.trace.record_event(
            phase='tool',
            event_type='mcp_tool_finish',
            payload={'tool_name': tool_name, 'result': summarize_text(raw_result, limit=800)},
        )
        payload = json.loads(raw_result)
        if not payload.get('ok', True):
            raise RuntimeError(payload.get('error', f'MCP tool failed: {tool_name}'))
        return payload

    def show_commit_patch(self, repo: str, commit_id: str) -> Dict[str, Any]:
        return self._call_tool('show_commit_patch', {'repo': repo, 'commit_id': commit_id})

    def resolve_parent_commit(self, repo: str, commit_id: str) -> str:
        payload = self._call_tool('resolve_parent_commit', {'repo': repo, 'commit_id': commit_id})
        return payload['parent_commit']

    def apply_check(self, repo: str, patch_path: str) -> tuple[bool, str]:
        payload = self._call_tool('apply_check', {'repo': repo, 'patch_path': patch_path})
        return bool(payload['apply_ok']), payload.get('stderr', '')

    def search_code(self,
                    repo: str,
                    pattern: str,
                    path_glob: Optional[str] = None,
                    mode: str = 'rg') -> List[CodeSearchHit]:
        payload = self._call_tool(
            'search_code',
            {
                'repo': repo,
                'pattern': pattern,
                'path_glob': path_glob,
                'mode': mode,
            },
        )
        return [CodeSearchHit.model_validate(item) for item in payload.get('hits', [])]

    def read_range(self, repo: str, path: str, start_line: int, end_line: int, reason: str) -> CodeSnippet:
        payload = self._call_tool(
            'read_range',
            {
                'repo': repo,
                'path': path,
                'start_line': start_line,
                'end_line': end_line,
            },
        )
        return CodeSnippet(
            file_path=payload['path'],
            start_line=payload['start_line'],
            end_line=payload['end_line'],
            content=payload['content'],
            reason=reason,
        )

    def find_symbol(self, repo: str, symbol_name: str, kind: str) -> Dict[str, Any]:
        return self._call_tool(
            'find_symbol',
            {
                'repo': repo,
                'symbol_name': symbol_name,
                'kind': kind,
            },
        )


def build_kernel_repo_mcp_config(server_name: str = 'kernel_repo') -> Dict[str, Any]:
    server_script = str((Path(__file__).resolve().parents[2] / 'scripts' / 'kernel_repo_mcp_server.py'))
    return {
        'mcpServers': {
            server_name: {
                'command': '/root/qwen-agent/.venv/bin/python',
                'args': [server_script],
            }
        }
    }


def get_kernel_repo_mcp_tools(server_name: str = 'kernel_repo',
                              allowed_suffixes: Optional[Sequence[str]] = None) -> List[Any]:
    tools = MCPManager().initConfig(build_kernel_repo_mcp_config(server_name))
    if not allowed_suffixes:
        return tools
    allowed_names = {f'{server_name}-{suffix}' for suffix in allowed_suffixes}
    return [tool for tool in tools if tool.name in allowed_names]


def _validate_agentic_tool_usage(responses: Sequence[Message], changed_files: Sequence[str], server_name: str) -> None:
    allowed_tools = {f'{server_name}-search_code', f'{server_name}-read_range'}
    changed_files = set(changed_files)
    for message in responses:
        if message.role != ASSISTANT or not message.function_call:
            continue
        tool_name = message.function_call.name
        if tool_name not in allowed_tools:
            raise RuntimeError(f'Agentic retriever called disallowed tool: {tool_name}')
        arguments = json5.loads(message.function_call.arguments)
        if tool_name.endswith('search_code'):
            path_glob = arguments.get('path_glob')
            if path_glob not in changed_files:
                raise RuntimeError(f'Agentic retriever search_code used path_glob outside changed_files: {path_glob}')
        elif tool_name.endswith('read_range'):
            path = arguments.get('path')
            if path not in changed_files:
                raise RuntimeError(f'Agentic retriever read_range used path outside changed_files: {path}')


def _count_agentic_tool_calls(responses: Sequence[Message]) -> int:
    return sum(1 for message in responses if message.role == ASSISTANT and message.function_call)


def run_agentic_retrieval(llm,
                          trace: TraceRecorder,
                          repo: str,
                          decoder_report: RootCauseReport,
                          origin_hunks: Sequence[PatchHunk],
                          changed_files: Sequence[str],
                          origin_patch: str,
                          server_name: str = 'kernel_repo',
                          max_tool_calls: int = 6,
                          timeout_sec: int = 90) -> RetrievalReport:
    system_message = (
        '你是 Linux 内核代码检索 Agent。必须通过 MCP 工具检索，不允许臆测。'
        '你必须只在 changed_files 范围内检索。'
        '前 2 次工具调用必须优先围绕 changed_files 和 impacted_functions。'
        '不要一开始先查新增 struct、macro、helper。'
        '只允许使用 search_code 和 read_range。最终只输出 JSON。'
    )
    prompt = build_agentic_retriever_prompt(
        decoder_report=decoder_report,
        origin_hunks=list(origin_hunks),
        changed_files=list(changed_files),
        origin_patch=origin_patch,
        repo=repo,
        max_tool_calls=max_tool_calls,
    )
    bot = FnCallAgent(
        llm=llm,
        system_message=system_message,
        function_list=get_kernel_repo_mcp_tools(server_name, allowed_suffixes=['search_code', 'read_range']),
        name='kernel_retriever',
    )

    class RetrieverTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise RetrieverTimeout(f'agentic retriever exceeded {timeout_sec}s')

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(timeout_sec)
        responses = bot.run_nonstream(messages=[Message(role='user', content=prompt)])
        signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    _validate_agentic_tool_usage(responses, changed_files=changed_files, server_name=server_name)
    tool_calls = _count_agentic_tool_calls(responses)
    if tool_calls > max_tool_calls:
        raise RuntimeError(f'Agentic retriever exceeded tool-call budget: {tool_calls} > {max_tool_calls}')
    final_text = ''
    for message in responses:
        if message.role == ASSISTANT and isinstance(message.content, str) and message.content.strip():
            final_text = message.content.strip()
    if not final_text:
        raise RuntimeError('Agentic retriever did not return final JSON output')
    trace.record_event(
        phase='retriever',
        event_type='agentic_retriever_output',
        payload={'summary': summarize_text(final_text, limit=1200)},
    )
    payload = json5.loads(final_text)
    return RetrievalReport.model_validate(payload)


def _extract_added_entities(origin_patch_text: str) -> Dict[str, List[str]]:
    macros = []
    structs = []
    functions = []
    includes = []
    function_re = re.compile(r'^(?:static\s+)?(?:inline\s+)?[\w\s\*]+\b([A-Za-z_]\w+)\s*\(')
    struct_re = re.compile(r'\bstruct\s+([A-Za-z_]\w+)\b')
    macro_re = re.compile(r'^#\s*define\s+([A-Za-z_]\w+)\b')
    include_re = re.compile(r'^#\s*include\s+([<"][^>"]+[>"])')

    for raw_line in origin_patch_text.splitlines():
        if not raw_line.startswith('+') or raw_line.startswith('+++'):
            continue
        line = raw_line[1:].strip()
        if not line:
            continue
        include_match = include_re.match(line)
        if include_match:
            includes.append(include_match.group(1))
        macro_match = macro_re.match(line)
        if macro_match:
            name = macro_match.group(1)
            if name not in macros:
                macros.append(name)
        for struct_name in struct_re.findall(line):
            if struct_name not in structs:
                structs.append(struct_name)
        fn_match = function_re.match(line)
        if fn_match:
            fn_name = fn_match.group(1)
            if fn_name not in ENTITY_BLACKLIST and fn_name not in functions:
                functions.append(fn_name)

    return {
        'added_macros': macros,
        'added_structs': structs,
        'added_functions': functions,
        'added_includes': includes,
    }


class CoverageDrivenRetriever:

    def __init__(self,
                 mcp_client: KernelRepoMCPClient,
                 repo: str,
                 origin_hunks: Sequence[PatchHunk],
                 semantic_anchors: Sequence[str],
                 path_hints: Sequence[str],
                 symbol_hints: Sequence[str],
                 origin_patch_text: str,
                 max_snippets: int = 12,
                 max_hits_per_anchor: int = 3):
        self.mcp_client = mcp_client
        self.repo = repo
        self.origin_hunks = list(origin_hunks)
        self.semantic_anchors = [anchor.strip() for anchor in semantic_anchors if anchor and anchor.strip()]
        self.path_hints = [path for path in path_hints if path and path.strip()]
        self.symbol_hints = [symbol.strip() for symbol in symbol_hints if symbol and symbol.strip()]
        self.max_snippets = max_snippets
        self.max_hits_per_anchor = max_hits_per_anchor
        self.added_entities = _extract_added_entities(origin_patch_text)
        self.snippets: List[CodeSnippet] = []
        self._seen_ranges = set()
        self._entity_cache: Dict[tuple[str, str, str], EntityCoverageReport] = {}

    def run(self) -> RetrievalReport:
        report = RetrievalReport(
            added_macros=self.added_entities['added_macros'],
            added_structs=self.added_entities['added_structs'],
            added_functions=self.added_entities['added_functions'],
            added_includes=self.added_entities['added_includes'],
        )

        for hunk_index, hunk in enumerate(self.origin_hunks):
            coverage = HunkCoverageReport(hunk_index=hunk_index, file_path=hunk.file_path)
            self._cover_hunk(hunk, coverage)
            report.hunk_coverages.append(coverage)

        entity_inputs = []
        entity_inputs.extend((name, 'function') for name in self.symbol_hints)
        entity_inputs.extend((name, 'macro') for name in self.added_entities['added_macros'])
        entity_inputs.extend((name, 'struct') for name in self.added_entities['added_structs'])
        entity_inputs.extend((name, 'function') for name in self.added_entities['added_functions'])
        seen_entities = set()
        for entity_name, entity_type in entity_inputs:
            key = (entity_name, entity_type)
            if key in seen_entities or not entity_name:
                continue
            seen_entities.add(key)
            report.entity_coverages.append(self._cover_entity(entity_name, entity_type))

        report.snippets = list(self.snippets)
        report.missing_entities = [
            coverage.entity_name for coverage in report.entity_coverages if not coverage.found
        ]
        return report

    def _append_snippet(self, snippet: CodeSnippet) -> bool:
        if len(self.snippets) >= self.max_snippets:
            return False
        key = (snippet.file_path, snippet.start_line, snippet.end_line)
        if key in self._seen_ranges:
            return False
        self._seen_ranges.add(key)
        self.snippets.append(snippet)
        return True

    def _capture_range(self,
                       file_path: str,
                       start_line: int,
                       end_line: int,
                       reason: str,
                       strategy: str,
                       coverage: Optional[HunkCoverageReport] = None,
                       entity_coverage: Optional[EntityCoverageReport] = None,
                       symbol_name: str = '',
                       hit_line: int = 0) -> bool:
        try:
            snippet = self.mcp_client.read_range(self.repo, file_path, start_line, end_line, reason=reason)
        except Exception as ex:
            target = coverage.notes if coverage is not None else entity_coverage.notes if entity_coverage else None
            if target is not None:
                target.append(f'{strategy} failed: {ex}')
            return False

        added = self._append_snippet(snippet)
        if not added:
            return False
        evidence = RetrievalEvidence(
            strategy=strategy,
            reason=reason,
            file_path=snippet.file_path,
            start_line=snippet.start_line,
            end_line=snippet.end_line,
            symbol_name=symbol_name,
            hit_line=hit_line,
        )
        if coverage is not None:
            coverage.evidence.append(evidence)
            coverage.covered = True
        if entity_coverage is not None:
            entity_coverage.evidence.append(evidence)
            entity_coverage.found = True
        return True

    def _cover_hunk(self, hunk: PatchHunk, coverage: HunkCoverageReport) -> None:
        self._capture_range(
            hunk.file_path,
            max(1, hunk.old_start - 20),
            hunk.old_start + max(hunk.old_count, 1) + 20,
            reason=f'hunk_context:{hunk.file_path}:{hunk.old_start}',
            strategy='direct_hunk_context',
            coverage=coverage,
        )

        for anchor in hunk.anchor_lines:
            anchor_hits = []
            try:
                anchor_hits = self.mcp_client.search_code(self.repo, anchor, path_glob=hunk.file_path)
            except Exception as ex:
                coverage.notes.append(f'anchor search failed for `{anchor}`: {ex}')
            if not anchor_hits:
                coverage.missing_anchors.append(anchor)
                continue
            for hit in anchor_hits[:self.max_hits_per_anchor]:
                self._capture_range(
                    hit.file_path,
                    max(1, hit.line_number - 20),
                    hit.line_number + 20,
                    reason=hit.reason,
                    strategy='anchor_match',
                    coverage=coverage,
                    hit_line=hit.line_number,
                )

        if not coverage.covered:
            try:
                self._capture_range(
                    hunk.file_path,
                    1,
                    160,
                    reason=f'path_bootstrap:{hunk.file_path}',
                    strategy='path_bootstrap',
                    coverage=coverage,
                )
            except Exception:
                coverage.notes.append(f'path bootstrap unavailable for {hunk.file_path}')

        if not coverage.covered:
            file_symbols = self._collect_symbols_for_file(hunk.file_path)
            coverage.notes.append('No same-file anchor hit found; relying on symbol and path-level evidence.')
            for symbol_name, symbol_kind in file_symbols:
                entity_coverage = self._cover_entity(symbol_name, symbol_kind, prefer_path=hunk.file_path, limit=1)
                if entity_coverage.found:
                    coverage.evidence.extend(entity_coverage.evidence[:1])
                    coverage.covered = True
                    break

    def _collect_symbols_for_file(self, file_path: str) -> List[tuple[str, str]]:
        symbols: List[tuple[str, str]] = []
        for name in self.symbol_hints:
            symbols.append((name, 'function'))
        for name in self.added_entities['added_macros']:
            symbols.append((name, 'macro'))
        for name in self.added_entities['added_structs']:
            symbols.append((name, 'struct'))
        for name in self.added_entities['added_functions']:
            symbols.append((name, 'function'))
        deduped = []
        seen = set()
        for name, kind in symbols:
            key = (name, kind)
            if not name or key in seen:
                continue
            seen.add(key)
            deduped.append((name, kind))
        return deduped

    def _cover_entity(self,
                      entity_name: str,
                      entity_type: str,
                      prefer_path: Optional[str] = None,
                      limit: int = 2) -> EntityCoverageReport:
        cache_key = (entity_name, entity_type, prefer_path or '')
        if cache_key in self._entity_cache:
            cached = self._entity_cache[cache_key]
            return EntityCoverageReport.model_validate(cached.model_dump())
        coverage = EntityCoverageReport(entity_name=entity_name, entity_type=entity_type)
        self._search_entity(entity_name, entity_type, coverage, prefer_path=prefer_path, limit=limit)
        if not coverage.found:
            coverage.notes.append('Entity not found; likely new in upstream or renamed in target tree.')
        self._entity_cache[cache_key] = EntityCoverageReport.model_validate(coverage.model_dump())
        return coverage

    def _search_entity(self,
                       entity_name: str,
                       entity_type: str,
                       coverage: EntityCoverageReport,
                       prefer_path: Optional[str],
                       limit: int) -> bool:
        kinds = [entity_type]
        if entity_type == 'function':
            kinds = ['function', 'global', 'macro']
        elif entity_type == 'global':
            kinds = ['global', 'function']

        found_count = 0
        for kind in kinds:
            try:
                payload = self.mcp_client.find_symbol(self.repo, entity_name, kind)
            except Exception as ex:
                coverage.notes.append(f'find_symbol failed for {entity_name}/{kind}: {ex}')
                continue

            candidates = payload.get('definitions', []) + payload.get('references', [])
            if prefer_path:
                preferred = [item for item in candidates if item.get('file_path') == prefer_path]
                others = [item for item in candidates if item.get('file_path') != prefer_path]
                candidates = preferred + others

            for item in candidates[:max(limit, self.max_hits_per_anchor)]:
                if self._capture_range(
                        item['file_path'],
                        max(1, int(item['line_number']) - 12),
                        int(item['line_number']) + 32,
                        reason=item['reason'],
                        strategy=f'entity_{kind}',
                        entity_coverage=coverage,
                        symbol_name=entity_name,
                        hit_line=int(item['line_number']),
                ):
                    found_count += 1
                if found_count >= limit:
                    return True

            # Fallback to literal search when symbol definitions are absent.
            try:
                hits = self.mcp_client.search_code(self.repo, entity_name, path_glob=prefer_path)
            except Exception as ex:
                coverage.notes.append(f'search_code fallback failed for {entity_name}: {ex}')
                hits = []

            for hit in hits[:limit]:
                if self._capture_range(
                        hit.file_path,
                        max(1, hit.line_number - 12),
                        hit.line_number + 32,
                        reason=hit.reason,
                        strategy='entity_literal_fallback',
                        entity_coverage=coverage,
                        symbol_name=entity_name,
                        hit_line=hit.line_number,
                ):
                    found_count += 1
                if found_count >= limit:
                    return True
        return coverage.found


def build_target_snippets_via_mcp(mcp_client: KernelRepoMCPClient,
                                  repo: str,
                                  origin_hunks: Iterable[PatchHunk],
                                  semantic_anchors: Iterable[str],
                                  path_hints: Iterable[str],
                                  symbol_hints: Iterable[str],
                                  origin_patch_text: str,
                                  max_snippets: int = 12,
                                  max_hits_per_anchor: int = 3) -> RetrievalReport:
    normalized_hunks = [hunk if isinstance(hunk, PatchHunk) else PatchHunk.model_validate(hunk) for hunk in origin_hunks]
    retriever = CoverageDrivenRetriever(
        mcp_client=mcp_client,
        repo=repo,
        origin_hunks=normalized_hunks,
        semantic_anchors=list(semantic_anchors),
        path_hints=list(path_hints),
        symbol_hints=list(symbol_hints),
        origin_patch_text=origin_patch_text,
        max_snippets=max_snippets,
        max_hits_per_anchor=max_hits_per_anchor,
    )
    report = retriever.run()

    if not report.snippets:
        # Last-resort fallback: try path bootstrap and semantic anchors globally.
        for path_hint in path_hints:
            try:
                snippet = mcp_client.read_range(repo, path_hint, 1, 160, reason=f'fallback_path:{path_hint}')
            except Exception:
                continue
            if snippet not in report.snippets:
                report.snippets.append(snippet)
            if len(report.snippets) >= max_snippets:
                break
        if len(report.snippets) < max_snippets:
            for anchor in semantic_anchors:
                if not anchor or not anchor.strip():
                    continue
                for hit in mcp_client.search_code(repo, anchor)[:max_hits_per_anchor]:
                    snippet = mcp_client.read_range(
                        repo,
                        hit.file_path,
                        max(1, hit.line_number - 20),
                        hit.line_number + 20,
                        reason=hit.reason,
                    )
                    report.snippets.append(snippet)
                    if len(report.snippets) >= max_snippets:
                        break
                if len(report.snippets) >= max_snippets:
                    break
    return report
