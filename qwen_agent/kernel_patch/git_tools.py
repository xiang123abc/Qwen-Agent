from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from qwen_agent.kernel_patch.models import CodeEdit, CodeSearchHit, CodeSnippet, PatchHunk, ToolTraceEntry
from qwen_agent.kernel_patch.trace import TraceRecorder

HUNK_RE = re.compile(r'^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@')


class GitCommandError(RuntimeError):

    def __init__(self, cmd: List[str], stderr: str, stdout: str = ''):
        self.cmd = cmd
        self.stderr = stderr
        self.stdout = stdout
        super().__init__(f'Command failed: {" ".join(cmd)}\n{stderr}')


def summarize_text(text: str, limit: int = 300) -> str:
    normalized = (text or '').strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + ' ...[truncated]'


class GitRepository:

    def __init__(self, path: str, trace: TraceRecorder, label: str):
        self.path = Path(path).expanduser().resolve()
        self.trace = trace
        self.label = label

    def run(self,
            args: List[str],
            *,
            check: bool = True,
            input_text: Optional[str] = None) -> subprocess.CompletedProcess:
        cmd = args
        self.trace.record_event(
            phase='tool',
            event_type='git_command_start',
            payload={
                'repo': str(self.path),
                'label': self.label,
                'cmd': cmd,
            },
        )
        proc = subprocess.run(
            cmd,
            cwd=self.path,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        self.trace.record_event(
            phase='tool',
            event_type='git_command_finish',
            payload={
                'repo': str(self.path),
                'label': self.label,
                'cmd': cmd,
                'returncode': proc.returncode,
                'stdout': summarize_text(proc.stdout),
                'stderr': summarize_text(proc.stderr),
            },
        )
        if check and proc.returncode != 0:
            raise GitCommandError(cmd=cmd, stderr=proc.stderr, stdout=proc.stdout)
        return proc

    def parent_commit(self, commit: str) -> str:
        return self.run(['git', 'rev-parse', f'{commit}^']).stdout.strip()

    def resolve_commit(self, commit: str) -> str:
        return self.run(['git', 'rev-parse', commit]).stdout.strip()

    def current_head(self) -> str:
        return self.run(['git', 'rev-parse', 'HEAD']).stdout.strip()

    def is_clean(self) -> bool:
        proc = self.run(['git', 'status', '--porcelain'], check=False)
        return not proc.stdout.strip()

    def commit_message(self, commit: str) -> str:
        return self.run(['git', 'show', '-s', '--format=%B', commit]).stdout.strip()

    def show_patch(self, commit: str) -> str:
        return self.run(['git', 'show', '-W', '--format=medium', '--patch', commit]).stdout

    def changed_files(self, commit: str) -> List[str]:
        output = self.run(['git', 'show', '--format=', '--name-only', commit]).stdout
        return [line.strip() for line in output.splitlines() if line.strip()]

    def file_exists(self, rel_path: str) -> bool:
        return (self.path / rel_path).exists()

    def search_fixed_string(self, pattern: str, path_glob: Optional[str] = None) -> List[CodeSearchHit]:
        if not pattern.strip():
            return []
        try:
            args = ['rg', '-n', '--no-heading', '--color', 'never', '-F', pattern]
            if path_glob:
                args.extend(['-g', path_glob])
            args.append('.')
            proc = self.run(args, check=False)
            if proc.returncode not in (0, 1):
                raise GitCommandError(cmd=args, stderr=proc.stderr, stdout=proc.stdout)
            output = proc.stdout
        except FileNotFoundError:
            args = ['git', 'grep', '-n', '-F', pattern]
            if path_glob:
                args.extend(['--', path_glob])
            proc = self.run(args, check=False)
            if proc.returncode not in (0, 1):
                raise GitCommandError(cmd=args, stderr=proc.stderr, stdout=proc.stdout)
            output = proc.stdout

        hits: List[CodeSearchHit] = []
        for line in output.splitlines():
            parts = line.split(':', 2)
            if len(parts) != 3:
                continue
            file_path, line_number, line_text = parts
            if file_path.startswith('./'):
                file_path = file_path[2:]
            hits.append(
                CodeSearchHit(
                    file_path=file_path,
                    line_number=int(line_number),
                    line_text=line_text,
                    reason=f'fixed_string:{pattern}',
                ))
        return hits

    def read_range(self, rel_path: str, start_line: int, end_line: int, reason: str) -> CodeSnippet:
        if start_line < 1:
            start_line = 1
        file_path = self.path / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f'{file_path} does not exist')
        cmd = ['bash', '-lc', f"nl -ba '{file_path}' | sed -n '{start_line},{end_line}p'"]
        proc = self.run(cmd)
        return CodeSnippet(
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            content=proc.stdout.rstrip(),
            reason=reason,
        )

    def apply_check(self, patch_path: str) -> Tuple[bool, str]:
        proc = self.run(['git', 'apply', '--check', patch_path], check=False)
        return proc.returncode == 0, proc.stderr.strip()

    def diff_check(self) -> Tuple[bool, str]:
        proc = self.run(['git', 'diff', '--check'], check=False)
        return proc.returncode == 0, proc.stderr.strip() or proc.stdout.strip()

    def diff(self, paths: Optional[List[str]] = None) -> str:
        args = ['git', 'diff', '--']
        if paths:
            args.extend(paths)
        return self.run(args).stdout

    def reset_hard(self, commit: str) -> None:
        self.run(['git', 'reset', '--hard', commit])

    def checkout_detached(self, commit: str) -> None:
        self.run(['git', 'checkout', '--detach', commit])

    def apply_code_edits(self, edits: List[CodeEdit]) -> None:
        grouped: Dict[str, List[CodeEdit]] = {}
        for edit in edits:
            grouped.setdefault(edit.file_path, []).append(edit)
        for file_path, file_edits in grouped.items():
            abs_path = self.path / file_path
            if not abs_path.exists():
                raise FileNotFoundError(f'{abs_path} does not exist')
            original_lines = abs_path.read_text(encoding='utf-8').splitlines(keepends=True)
            file_edits = sorted(file_edits, key=lambda item: item.start_line, reverse=True)
            for edit in file_edits:
                start = max(1, edit.start_line)
                end = max(start, edit.end_line)
                replacement = edit.new_content
                replacement_lines = [] if replacement == '' else replacement.splitlines(keepends=True)
                if replacement and not replacement.endswith('\n'):
                    replacement_lines[-1] = replacement_lines[-1] + '\n'
                original_lines[start - 1:end] = replacement_lines
            abs_path.write_text(''.join(original_lines), encoding='utf-8')


def prepare_worktree(base_repo: GitRepository, target_commit: str, worktree_path: Path) -> Path:
    if (worktree_path / '.git').exists():
        return worktree_path
    if worktree_path.exists() and any(worktree_path.iterdir()):
        raise FileExistsError(f'Worktree path already exists and is not empty: {worktree_path}')
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    base_repo.run(['git', 'worktree', 'add', '--detach', str(worktree_path), target_commit])
    return worktree_path


def parse_patch_hunks(origin_patch: str, max_anchor_lines: int = 8) -> List[PatchHunk]:
    hunks: List[PatchHunk] = []
    current_file: Optional[str] = None
    current_hunk: Optional[PatchHunk] = None
    seen_anchors: Dict[Tuple[str, int], set] = {}
    for raw_line in origin_patch.splitlines():
        if raw_line.startswith('diff --git '):
            match = re.match(r'diff --git a/(.+) b/(.+)', raw_line)
            if match:
                current_file = match.group(2)
            current_hunk = None
            continue
        match = HUNK_RE.match(raw_line)
        if match and current_file:
            current_hunk = PatchHunk(
                file_path=current_file,
                old_start=int(match.group('old_start')),
                old_count=int(match.group('old_count') or '1'),
                new_start=int(match.group('new_start')),
                new_count=int(match.group('new_count') or '1'),
                anchor_lines=[],
            )
            hunks.append(current_hunk)
            seen_anchors[(current_file, len(hunks) - 1)] = set()
            continue
        if not current_hunk or not raw_line:
            continue
        if raw_line[0] not in (' ', '-'):
            continue
        candidate = raw_line[1:].strip()
        if not candidate or len(candidate) < 4:
            continue
        if candidate in ('{', '}', 'break;', 'return;', 'else', 'if', '#endif', '#else'):
            continue
        key = (current_hunk.file_path, hunks.index(current_hunk))
        if len(current_hunk.anchor_lines) >= max_anchor_lines:
            continue
        if candidate in seen_anchors[key]:
            continue
        seen_anchors[key].add(candidate)
        current_hunk.anchor_lines.append(candidate)
    return hunks


def build_target_snippets(target_repo: GitRepository,
                          origin_hunks: Iterable[PatchHunk],
                          semantic_anchors: Iterable[str],
                          path_hints: Iterable[str],
                          max_snippets: int = 8,
                          max_hits_per_anchor: int = 2) -> List[CodeSnippet]:
    snippets: List[CodeSnippet] = []
    seen_ranges = set()

    def append_snippet(snippet: CodeSnippet) -> bool:
        snippets.append(snippet)
        return len(snippets) >= max_snippets

    for hunk in origin_hunks:
        if target_repo.file_exists(hunk.file_path):
            start = max(1, hunk.old_start - 20)
            end = hunk.old_start + max(hunk.old_count, 1) + 20
            key = (hunk.file_path, start, end)
            if key not in seen_ranges:
                seen_ranges.add(key)
                if append_snippet(
                        target_repo.read_range(
                            hunk.file_path,
                            start,
                            end,
                            reason=f'hunk_context:{hunk.file_path}:{hunk.old_start}',
                        )):
                    return snippets
        for anchor in hunk.anchor_lines:
            hit_count = 0
            for hit in target_repo.search_fixed_string(anchor, path_glob=hunk.file_path):
                start = max(1, hit.line_number - 20)
                end = hit.line_number + 20
                key = (hit.file_path, start, end)
                if key in seen_ranges:
                    continue
                seen_ranges.add(key)
                hit_count += 1
                if append_snippet(target_repo.read_range(hit.file_path, start, end, reason=hit.reason)):
                    return snippets
                if hit_count >= max_hits_per_anchor:
                    break
    for path_hint in path_hints:
        if target_repo.file_exists(path_hint):
            key = (path_hint, 1, 120)
            if key not in seen_ranges:
                seen_ranges.add(key)
                if append_snippet(target_repo.read_range(path_hint, 1, 120, reason=f'path_hint:{path_hint}')):
                    return snippets
    for anchor in semantic_anchors:
        hit_count = 0
        for hit in target_repo.search_fixed_string(anchor):
            start = max(1, hit.line_number - 20)
            end = hit.line_number + 20
            key = (hit.file_path, start, end)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            hit_count += 1
            if append_snippet(target_repo.read_range(hit.file_path, start, end, reason=hit.reason)):
                return snippets
            if hit_count >= max_hits_per_anchor:
                break
    return snippets
