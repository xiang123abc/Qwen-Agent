#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


server = FastMCP('kernel_repo')

HUNK_RE = re.compile(r'^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@')


class SearchHit(BaseModel):
    file_path: str
    line_number: int
    line_text: str
    reason: str


def run_in_repo(repo: str, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, cwd=repo, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f'command failed: {args}')
    return proc


def resolve_repo_path(repo: str, path: str) -> Path:
    repo_root = Path(repo).resolve()
    target = (repo_root / path).resolve()
    if target != repo_root and repo_root not in target.parents:
        raise RuntimeError(f'Path escapes repo root: {path}')
    return target


def parse_patch_hunks(origin_patch: str, max_anchor_lines: int = 8) -> list[dict]:
    hunks: list[dict] = []
    current_file: Optional[str] = None
    current_hunk: Optional[dict] = None
    for raw_line in origin_patch.splitlines():
        if raw_line.startswith('diff --git '):
            match = re.match(r'diff --git a/(.+) b/(.+)', raw_line)
            if match:
                current_file = match.group(2)
            current_hunk = None
            continue
        match = HUNK_RE.match(raw_line)
        if match and current_file:
            current_hunk = {
                'file_path': current_file,
                'old_start': int(match.group('old_start')),
                'old_count': int(match.group('old_count') or '1'),
                'new_start': int(match.group('new_start')),
                'new_count': int(match.group('new_count') or '1'),
                'anchor_lines': [],
            }
            hunks.append(current_hunk)
            continue
        if not current_hunk or not raw_line or raw_line[0] not in (' ', '-'):
            continue
        candidate = raw_line[1:].strip()
        if not candidate or len(candidate) < 4:
            continue
        if len(current_hunk['anchor_lines']) >= max_anchor_lines:
            continue
        if candidate in current_hunk['anchor_lines']:
            continue
        current_hunk['anchor_lines'].append(candidate)
    return hunks


@server.tool(description='Search code in a repository using rg or git grep and return hits with line numbers.')
def search_code(repo: str, pattern: str, path_glob: str | None = None,
                mode: Literal['rg', 'git_grep'] = 'rg') -> str:
    try:
        if not pattern.strip():
            return json.dumps({'ok': True, 'hits': []}, ensure_ascii=False)
        if mode == 'rg':
            args = ['rg', '-n', '--no-heading', '--color', 'never', '-F', pattern]
            if path_glob:
                args.extend(['-g', path_glob])
            args.append('.')
        else:
            args = ['git', 'grep', '-n', '-F', pattern]
            if path_glob:
                args.extend(['--', path_glob])
        proc = run_in_repo(repo, args, check=False)
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        hits = []
        for line in proc.stdout.splitlines():
            parts = line.split(':', 2)
            if len(parts) != 3:
                continue
            file_path, line_number, line_text = parts
            if file_path.startswith('./'):
                file_path = file_path[2:]
            hits.append(
                SearchHit(
                    file_path=file_path,
                    line_number=int(line_number),
                    line_text=line_text,
                    reason=f'{mode}:{pattern}',
                ).model_dump())
        return json.dumps({'ok': True, 'hits': hits}, ensure_ascii=False)
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Read a specific line range from a file in the repository.')
def read_range(repo: str, path: str, start_line: int, end_line: int) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        if start_line < 1:
            start_line = 1
        cmd = ['bash', '-lc', f"nl -ba '{file_path}' | sed -n '{start_line},{end_line}p'"]
        proc = run_in_repo(repo, cmd)
        return json.dumps(
            {
                'ok': True,
                'path': path,
                'start_line': start_line,
                'end_line': end_line,
                'content': proc.stdout.rstrip(),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Read a full file from the repository.')
def read_file(repo: str, path: str) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        return json.dumps(
            {
                'ok': True,
                'path': path,
                'content': file_path.read_text(encoding='utf-8'),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Write a full file in the repository.')
def write_file(repo: str, path: str, content: str) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return json.dumps(
            {
                'ok': True,
                'path': path,
                'bytes_written': len(content.encode('utf-8')),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Replace a literal text block inside a repository file.')
def replace_in_file(repo: str, path: str, old_text: str, new_text: str, expected_count: int = 1) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        content = file_path.read_text(encoding='utf-8')
        occurrences = content.count(old_text)
        if occurrences != expected_count:
            raise RuntimeError(f'Expected {expected_count} occurrences, found {occurrences}')
        updated = content.replace(old_text, new_text, expected_count)
        file_path.write_text(updated, encoding='utf-8')
        return json.dumps(
            {
                'ok': True,
                'path': path,
                'occurrences': occurrences,
                'bytes_written': len(updated.encode('utf-8')),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Replace an inclusive line range inside a repository file.')
def replace_lines(repo: str, path: str, start_line: int, end_line: int, new_text: str) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        replacement = [] if new_text == '' else new_text.splitlines(keepends=True)
        if new_text and not new_text.endswith('\n'):
            replacement[-1] = replacement[-1] + '\n'
        lines[start - 1:end] = replacement
        file_path.write_text(''.join(lines), encoding='utf-8')
        return json.dumps(
            {
                'ok': True,
                'path': path,
                'start_line': start,
                'end_line': end,
                'bytes_written': len(''.join(lines).encode('utf-8')),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Insert text before a literal anchor in a repository file.')
def insert_before(repo: str, path: str, anchor_text: str, new_text: str, expected_count: int = 1) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        content = file_path.read_text(encoding='utf-8')
        occurrences = content.count(anchor_text)
        if occurrences != expected_count:
            raise RuntimeError(f'Expected {expected_count} occurrences, found {occurrences}')
        replacement = new_text + anchor_text
        updated = content.replace(anchor_text, replacement, expected_count)
        file_path.write_text(updated, encoding='utf-8')
        return json.dumps({'ok': True, 'path': path, 'occurrences': occurrences}, ensure_ascii=False)
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Insert text after a literal anchor in a repository file.')
def insert_after(repo: str, path: str, anchor_text: str, new_text: str, expected_count: int = 1) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        content = file_path.read_text(encoding='utf-8')
        occurrences = content.count(anchor_text)
        if occurrences != expected_count:
            raise RuntimeError(f'Expected {expected_count} occurrences, found {occurrences}')
        replacement = anchor_text + new_text
        updated = content.replace(anchor_text, replacement, expected_count)
        file_path.write_text(updated, encoding='utf-8')
        return json.dumps({'ok': True, 'path': path, 'occurrences': occurrences}, ensure_ascii=False)
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Replace a line range around the first matching anchor line in a repository file.')
def replace_near_anchor(repo: str,
                        path: str,
                        anchor_text: str,
                        start_offset: int,
                        end_offset: int,
                        new_text: str,
                        expected_anchor_count: int = 1) -> str:
    try:
        file_path = resolve_repo_path(repo, path)
        lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)
        matched_indices = [idx for idx, line in enumerate(lines) if anchor_text in line]
        if len(matched_indices) != expected_anchor_count:
            raise RuntimeError(f'Expected {expected_anchor_count} anchor matches, found {len(matched_indices)}')
        anchor_idx = matched_indices[0]
        start = max(0, anchor_idx + int(start_offset))
        end = min(len(lines), anchor_idx + int(end_offset) + 1)
        replacement = [] if new_text == '' else new_text.splitlines(keepends=True)
        if new_text and not new_text.endswith('\n'):
            replacement[-1] = replacement[-1] + '\n'
        lines[start:end] = replacement
        file_path.write_text(''.join(lines), encoding='utf-8')
        return json.dumps(
            {
                'ok': True,
                'path': path,
                'anchor_line': anchor_idx + 1,
                'start_line': start + 1,
                'end_line': end,
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='List files in the repository using rg --files or find.')
def list_files(repo: str, path_glob: str | None = None) -> str:
    try:
        if path_glob:
            proc = run_in_repo(repo, ['rg', '--files', '-g', path_glob], check=False)
            if proc.returncode not in (0, 1):
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
            files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        else:
            proc = run_in_repo(repo, ['rg', '--files'], check=False)
            if proc.returncode not in (0, 1):
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
            files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return json.dumps({'ok': True, 'files': files[:500]}, ensure_ascii=False)
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Show a commit patch with commit message, changed files, and parsed hunks.')
def show_commit_patch(repo: str, commit_id: str) -> str:
    try:
        patch_text = run_in_repo(repo, ['git', 'show', '-W', '--format=medium', '--patch', commit_id]).stdout
        changed_files = run_in_repo(repo, ['git', 'show', '--format=', '--name-only', commit_id]).stdout
        commit_message = run_in_repo(repo, ['git', 'show', '-s', '--format=%B', commit_id]).stdout
        return json.dumps(
            {
                'ok': True,
                'commit_id': commit_id,
                'commit_message': commit_message.strip(),
                'changed_files': [line.strip() for line in changed_files.splitlines() if line.strip()],
                'patch_text': patch_text,
                'hunks': parse_patch_hunks(patch_text),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Resolve the parent commit of a given commit id inside a repository.')
def resolve_parent_commit(repo: str, commit_id: str) -> str:
    try:
        parent_commit = run_in_repo(repo, ['git', 'rev-parse', f'{commit_id}^']).stdout.strip()
        return json.dumps(
            {
                'ok': True,
                'repo': repo,
                'commit_id': commit_id,
                'parent_commit': parent_commit,
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Run git apply --check for a patch file against a repository.')
def apply_check(repo: str, patch_path: str) -> str:
    try:
        proc = run_in_repo(repo, ['git', 'apply', '--check', patch_path], check=False)
        return json.dumps(
            {
                'ok': True,
                'repo': repo,
                'patch_path': patch_path,
                'apply_ok': proc.returncode == 0,
                'stderr': proc.stderr.strip(),
                'stdout': proc.stdout.strip(),
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Run a git command inside the repository.')
def run_git(repo: str, args: List[str]) -> str:
    try:
        if not args:
            raise RuntimeError('args must not be empty')
        proc = run_in_repo(repo, ['git'] + args, check=False)
        return json.dumps(
            {
                'ok': True,
                'args': args,
                'returncode': proc.returncode,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Run a shell command inside the repository.')
def run_command(repo: str, command: str, timeout_sec: int = 30) -> str:
    try:
        proc = subprocess.run(['bash', '-lc', command],
                              cwd=repo,
                              text=True,
                              capture_output=True,
                              check=False,
                              timeout=max(1, timeout_sec),
                              env=os.environ.copy())
        return json.dumps(
            {
                'ok': True,
                'command': command,
                'returncode': proc.returncode,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


@server.tool(description='Find likely definitions and references of a C symbol in a repository.')
def find_symbol(repo: str, symbol_name: str, kind: Literal['function', 'struct', 'macro', 'global']) -> str:
    try:
        if kind == 'function':
            definition_pattern = rf'^\s*(?:static\s+)?(?:inline\s+)?[\w\s\*\(\),]+\b{re.escape(symbol_name)}\s*\('
        elif kind == 'struct':
            definition_pattern = rf'^\s*struct\s+{re.escape(symbol_name)}\b'
        elif kind == 'macro':
            definition_pattern = rf'^\s*#\s*define\s+{re.escape(symbol_name)}\b'
        else:
            definition_pattern = rf'^\s*(?:extern\s+)?[\w\s\*]+\b{re.escape(symbol_name)}\b(?:\s*[;=])'

        def_proc = run_in_repo(repo, ['rg', '-n', '--no-heading', '-e', definition_pattern, '.'], check=False)
        ref_proc = run_in_repo(repo, ['rg', '-n', '--no-heading', '-F', symbol_name, '.'], check=False)

        def parse_hits(output: str, reason: str):
            hits = []
            for line in output.splitlines():
                parts = line.split(':', 2)
                if len(parts) != 3:
                    continue
                file_path, line_number, line_text = parts
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                hits.append(
                    {
                        'file_path': file_path,
                        'line_number': int(line_number),
                        'line_text': line_text,
                        'reason': reason,
                    })
            return hits

        definitions = parse_hits(def_proc.stdout, f'{kind}_definition:{symbol_name}')
        references = parse_hits(ref_proc.stdout, f'{kind}_reference:{symbol_name}')
        return json.dumps(
            {
                'ok': True,
                'symbol_name': symbol_name,
                'kind': kind,
                'definitions': definitions,
                'references': references[:20],
            },
            ensure_ascii=False,
        )
    except Exception as ex:
        return json.dumps({'ok': False, 'error': str(ex)}, ensure_ascii=False)


if __name__ == '__main__':
    server.run(transport='stdio')
