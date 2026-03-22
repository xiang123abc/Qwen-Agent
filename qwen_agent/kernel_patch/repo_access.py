"""面向 Linux 大仓的局部代码读取与片段匹配层。

该模块模拟 MCP 风格的仓库读取能力：
- 文本检索；
- 局部范围读取；
- revision 文件读取；
- 基于片段指纹的局部匹配；
- 基于前后锚点的插入点解析。
"""

from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from qwen_agent.utils.utils import read_text_from_file

from .autopatch_models import CodeWindow, InsertionAnchor, SearchHit, SnippetMatch
from .git_ops import KernelRepoManager

LINE_NUMBER_PREFIX_RE = re.compile(r'^\s*\d+\s+')


def _strip_line_number_prefix(text: str) -> str:
    return '\n'.join(LINE_NUMBER_PREFIX_RE.sub('', line) for line in text.splitlines())


def _normalize_line(line: str) -> str:
    line = LINE_NUMBER_PREFIX_RE.sub('', line)
    line = line.rstrip()
    if line[:1] in ('+', '-', ' '):
        line = line[1:]
    line = line.strip()
    line = re.sub(r'//.*$', '', line)
    line = re.sub(r'/\*.*?\*/', '', line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()


def _meaningful_lines(text: str) -> List[str]:
    return [normalized for raw in text.splitlines() if (normalized := _normalize_line(raw))]


def _window_similarity(expected: Sequence[str], actual: Sequence[str]) -> float:
    if not expected or not actual:
        return 0.0
    return SequenceMatcher(None, '\n'.join(expected), '\n'.join(actual)).ratio()


class LocalRepoMCPClient:
    """本地仓库访问层，提供 MCP 风格的最小接口。"""

    def __init__(self, manager: KernelRepoManager):
        self.manager = manager

    def read_text(self, worktree_path: Path, relative_path: str) -> str:
        full_path = (worktree_path / relative_path).resolve()
        return read_text_from_file(str(full_path))

    def read_range(self,
                   worktree_path: Path,
                   relative_path: str,
                   start_line: int,
                   end_line: int,
                   reason: str = '') -> CodeWindow:
        content = self.manager.read_file_slice(worktree_path, relative_path, start_line, end_line)
        return CodeWindow(path=relative_path,
                          start_line=max(1, start_line),
                          end_line=max(start_line, end_line),
                          reason=reason,
                          content=content)

    def read_file_head(self, worktree_path: Path, relative_path: str, max_lines: int = 80, reason: str = '') -> CodeWindow:
        return self.read_range(worktree_path, relative_path, 1, max_lines, reason=reason or 'file_head')

    def read_revision_text(self, revision: str, relative_path: str) -> str:
        return self.manager.read_revision_file(revision, relative_path)

    def git_log_pickaxe(self, token: str, paths: Optional[Sequence[str]] = None, limit: int = 8) -> str:
        cmd = ['log', '--no-merges', f'-S{token}', f'--format=%H %s', '-n', str(limit)]
        if paths:
            cmd.append('--')
            cmd.extend(paths)
        return self.manager.git(*cmd, timeout=120).strip()

    def search_text(self,
                    worktree_path: Path,
                    pattern: str,
                    paths: Optional[Sequence[str]] = None,
                    max_results: int = 40,
                    term: str = '') -> List[SearchHit]:
        raw = self.manager.search_code(worktree_path, pattern=pattern, paths=paths, max_results=max_results)
        if raw == 'No matches found.':
            return []
        hits: List[SearchHit] = []
        for line in raw.splitlines():
            if line.startswith('... truncated'):
                break
            parts = line.split(':', 2)
            if len(parts) != 3:
                continue
            path, line_no, preview = parts
            try:
                parsed_line = int(line_no)
            except ValueError:
                continue
            hits.append(SearchHit(path=path, line_no=parsed_line, preview=preview.strip(), term=term))
        return hits

    def search_literal(self,
                       worktree_path: Path,
                       literal: str,
                       paths: Optional[Sequence[str]] = None,
                       max_results: int = 40) -> List[SearchHit]:
        escaped = re.escape(literal.strip())
        return self.search_text(worktree_path, escaped, paths=paths, max_results=max_results, term=literal)

    def locate_snippet(self,
                       worktree_path: Path,
                       relative_path: str,
                       snippet: str,
                       max_candidates: int = 5) -> List[SnippetMatch]:
        if not snippet.strip():
            return []

        file_lines = self.read_text(worktree_path, relative_path).splitlines()
        snippet_lines = _meaningful_lines(snippet)
        if not snippet_lines:
            return []

        exact_matches: List[SnippetMatch] = []
        for start in range(0, len(file_lines) - len(snippet_lines) + 1):
            candidate = [_normalize_line(line) for line in file_lines[start:start + len(snippet_lines)]]
            if candidate == snippet_lines:
                matched_text = '\n'.join(file_lines[start:start + len(snippet_lines)])
                exact_matches.append(
                    SnippetMatch(path=relative_path,
                                 start_line=start + 1,
                                 end_line=start + len(snippet_lines),
                                 score=1.0,
                                 matched_text=matched_text,
                                 strategy='exact'))
        if exact_matches:
            return exact_matches[:max_candidates]

        results: List[SnippetMatch] = []
        min_window = max(1, len(snippet_lines) - 2)
        max_window = min(len(file_lines), len(snippet_lines) + 6)
        for start in range(len(file_lines)):
            for size in range(min_window, max_window + 1):
                end = start + size
                if end > len(file_lines):
                    break
                actual_lines = [_normalize_line(line) for line in file_lines[start:end]]
                actual_lines = [line for line in actual_lines if line]
                score = _window_similarity(snippet_lines, actual_lines)
                if score < 0.60:
                    continue
                matched_text = '\n'.join(file_lines[start:end])
                results.append(
                    SnippetMatch(path=relative_path,
                                 start_line=start + 1,
                                 end_line=end,
                                 score=score,
                                 matched_text=matched_text,
                                 strategy='fuzzy'))
        results.sort(key=lambda item: (-item.score, item.start_line, item.end_line))
        deduped: List[SnippetMatch] = []
        seen = set()
        for item in results:
            key = (item.start_line, item.end_line)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_candidates:
                break
        return deduped

    def resolve_insertion_point(self,
                                worktree_path: Path,
                                relative_path: str,
                                before_snippet: str = '',
                                after_snippet: str = '') -> Optional[InsertionAnchor]:
        before_matches = self.locate_snippet(worktree_path, relative_path, before_snippet, max_candidates=1)
        after_matches = self.locate_snippet(worktree_path, relative_path, after_snippet, max_candidates=1)

        before = before_matches[0] if before_matches else None
        after = after_matches[0] if after_matches else None
        if before and after and before.end_line < after.start_line:
            score = min(before.score, after.score)
            return InsertionAnchor(path=relative_path,
                                   line_no=before.end_line + 1,
                                   score=score,
                                   strategy='between_anchors',
                                   before_line=before.end_line,
                                   after_line=after.start_line)
        if before:
            return InsertionAnchor(path=relative_path,
                                   line_no=before.end_line + 1,
                                   score=before.score,
                                   strategy='after_before_anchor',
                                   before_line=before.end_line)
        if after:
            return InsertionAnchor(path=relative_path,
                                   line_no=max(1, after.start_line),
                                   score=after.score,
                                   strategy='before_after_anchor',
                                   after_line=after.start_line)
        return None

    def surrounding_window(self,
                           worktree_path: Path,
                           relative_path: str,
                           line_no: int,
                           radius: int = 30,
                           reason: str = '') -> CodeWindow:
        start = max(1, line_no - radius)
        end = line_no + radius
        return self.read_range(worktree_path, relative_path, start, end, reason=reason or f'around:{line_no}')

    def locate_snippet_in_paths(self,
                                worktree_path: Path,
                                paths: Iterable[str],
                                snippet: str,
                                max_candidates: int = 8) -> List[SnippetMatch]:
        matches: List[SnippetMatch] = []
        for path in paths:
            matches.extend(self.locate_snippet(worktree_path, path, snippet, max_candidates=max_candidates))
        matches.sort(key=lambda item: (-item.score, item.path, item.start_line))
        return matches[:max_candidates]

    def strip_numbered_window(self, window: CodeWindow) -> str:
        return _strip_line_number_prefix(window.content)
