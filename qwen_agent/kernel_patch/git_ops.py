import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from qwen_agent.log import logger
from qwen_agent.utils.utils import read_text_from_file, save_text_to_file

from .models import KernelCaseBundle, PatchCase, truncate_middle


class GitCommandError(RuntimeError):
    pass


def _ensure_within(root: Path, candidate: Path) -> None:
    root = root.resolve()
    candidate = candidate.resolve()
    if os.path.commonpath([str(root), str(candidate)]) != str(root):
        raise ValueError(f'Path `{candidate}` escapes repository root `{root}`')


class KernelRepoManager:

    def __init__(self, repo_root: str, workspace_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.worktree_root = self.workspace_root / 'worktrees'
        self.artifacts_root = self.workspace_root / 'artifacts'
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        if not (self.repo_root / '.git').exists():
            raise ValueError(f'Not a git repository: {self.repo_root}')

    def _run(self,
             cmd: Sequence[str],
             cwd: Optional[Path] = None,
             timeout: int = 60,
             check: bool = True) -> subprocess.CompletedProcess:
        process = subprocess.run(cmd,
                                 cwd=str(cwd or self.repo_root),
                                 capture_output=True,
                                 text=True,
                                 encoding='utf-8',
                                 errors='replace',
                                 timeout=timeout)
        if check and process.returncode != 0:
            raise GitCommandError(f'Command failed: {" ".join(cmd)}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}')
        return process

    def git(self, *args: str, cwd: Optional[Path] = None, timeout: int = 60) -> str:
        return self._run(['git', *args], cwd=cwd, timeout=timeout).stdout

    def rev_parse(self, revision: str) -> str:
        return self.git('rev-parse', revision).strip()

    def parent_commit(self, commit: str) -> str:
        return self.rev_parse(f'{commit}^')

    def changed_files(self, commit: str) -> List[str]:
        output = self.git('diff-tree', '--no-commit-id', '--name-only', '-r', commit)
        return [line.strip() for line in output.splitlines() if line.strip()]

    def commit_subject(self, commit: str) -> str:
        return self.git('log', '--format=%s', '-n', '1', commit).strip()

    def commit_message(self, commit: str) -> str:
        return self.git('log', '--format=%B', '-n', '1', commit).strip()

    def diff_stat(self, base_commit: str, fix_commit: str) -> str:
        return self.git('diff', '--stat', base_commit, fix_commit).strip()

    def reference_show(self, fix_commit: str, paths: Optional[Iterable[str]] = None) -> str:
        cmd = ['show', '--patch', '--no-ext-diff', '--format=fuller', '--stat', '-W', fix_commit]
        if paths:
            cmd.append('--')
            cmd.extend(paths)
        return self.git(*cmd, timeout=120)

    def community_patch(self, base_commit: str, fix_commit: str) -> str:
        return self.git('diff', '--binary', '--no-ext-diff', base_commit, fix_commit, timeout=120)

    def prepare_case_bundle(self, case: PatchCase, reference_char_limit: int = 50000) -> KernelCaseBundle:
        fix_commit = self.rev_parse(case.fix_commit)
        base_commit = self.parent_commit(fix_commit)
        changed_files = self.changed_files(fix_commit)
        reference_show_text = self.reference_show(fix_commit, paths=changed_files or None)
        community_patch_text = self.community_patch(base_commit, fix_commit)
        bundle = KernelCaseBundle(case=case,
                                  repo_root=self.repo_root,
                                  workspace_root=self.workspace_root,
                                  base_commit=base_commit,
                                  fix_commit=fix_commit,
                                  commit_subject=self.commit_subject(fix_commit),
                                  commit_message=self.commit_message(fix_commit),
                                  changed_files=changed_files,
                                  diff_stat=self.diff_stat(base_commit, fix_commit),
                                  reference_show_excerpt=truncate_middle(reference_show_text, reference_char_limit),
                                  community_patch=community_patch_text,
                                  community_patch_excerpt=truncate_middle(community_patch_text, reference_char_limit))
        bundle.artifact_dir.mkdir(parents=True, exist_ok=True)
        return bundle

    def symbol_context(self,
                       worktree_path: Path,
                       symbol: str,
                       paths: Optional[Sequence[str]] = None,
                       context_before: int = 20,
                       context_after: int = 80,
                       max_hits: int = 3) -> str:
        hits = self.search_code(worktree_path, pattern=symbol, paths=paths, max_results=max_hits)
        if hits == 'No matches found.':
            return hits

        sections = []
        for raw_hit in hits.splitlines():
            if raw_hit.startswith('... truncated'):
                break
            parts = raw_hit.split(':', 2)
            if len(parts) < 3:
                continue
            file_path, line_no_str, _ = parts
            try:
                line_no = int(line_no_str)
            except ValueError:
                continue
            start_line = max(1, line_no - context_before)
            end_line = line_no + context_after
            snippet = self.read_file_slice(worktree_path, file_path, start_line, end_line)
            sections.append(f'## {file_path}:{line_no}\n{snippet}')
        return '\n\n'.join(sections) if sections else 'No matches found.'

    def current_hunk_context(self,
                             bundle: KernelCaseBundle,
                             worktree_path: Path,
                             context_before: int = 12,
                             context_after: int = 50,
                             max_hunks_per_file: int = 3) -> str:
        current_file = None
        hunks_per_file = {}
        sections = []
        hunk_re = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
        for line in bundle.community_patch.splitlines():
            if line.startswith('diff --git '):
                parts = line.split()
                if len(parts) >= 4:
                    path = parts[2]
                    if path.startswith('a/'):
                        path = path[2:]
                    current_file = path
                else:
                    current_file = None
                continue

            if current_file is None:
                continue

            match = hunk_re.match(line)
            if not match:
                continue

            hunks_per_file[current_file] = hunks_per_file.get(current_file, 0) + 1
            if hunks_per_file[current_file] > max_hunks_per_file:
                continue

            old_start = int(match.group(1))
            start_line = max(1, old_start - context_before)
            end_line = old_start + context_after
            try:
                snippet = self.read_file_slice(worktree_path, current_file, start_line, end_line)
            except Exception:
                continue
            sections.append(f'## Current parent code: {current_file}:{old_start}\n{snippet}')

        return '\n\n'.join(sections) if sections else 'No current hunk context available.'

    def worktree_path(self, bundle: KernelCaseBundle) -> Path:
        return self.worktree_root / bundle.case.slug

    def remove_worktree(self, worktree_path: Path) -> None:
        if worktree_path.exists():
            try:
                self._run(['git', 'worktree', 'remove', '--force', str(worktree_path)],
                          cwd=self.repo_root,
                          timeout=120,
                          check=False)
            finally:
                shutil.rmtree(worktree_path, ignore_errors=True)
        self._run(['git', 'worktree', 'prune'], cwd=self.repo_root, timeout=120, check=False)

    def create_worktree(self, bundle: KernelCaseBundle, recreate: bool = False) -> Path:
        worktree_path = self.worktree_path(bundle)
        if recreate:
            self.remove_worktree(worktree_path)
        if not worktree_path.exists():
            logger.info(f'Creating worktree {worktree_path} at {bundle.base_commit}')
            self._run(['git', 'worktree', 'add', '--detach', str(worktree_path), bundle.base_commit],
                      cwd=self.repo_root,
                      timeout=240)
        self.reset_worktree(worktree_path, bundle.base_commit)
        return worktree_path

    def reset_worktree(self, worktree_path: Path, base_commit: str) -> None:
        if not worktree_path.exists():
            raise ValueError(f'Worktree does not exist: {worktree_path}')
        self.git('checkout', '--detach', base_commit, cwd=worktree_path, timeout=120)
        self.git('reset', '--hard', base_commit, cwd=worktree_path, timeout=120)
        self.git('clean', '-fdq', cwd=worktree_path, timeout=120)

    def read_file_slice(self, worktree_path: Path, relative_path: str, start_line: int, end_line: int) -> str:
        full_path = (worktree_path / relative_path).resolve()
        _ensure_within(worktree_path, full_path)
        if not full_path.exists():
            raise ValueError(f'File does not exist: {relative_path}')
        if end_line < start_line:
            raise ValueError('end_line must be >= start_line')
        lines = read_text_from_file(str(full_path)).splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        selected = []
        for line_no in range(start, end + 1):
            selected.append(f'{line_no:6d} {lines[line_no - 1]}')
        return '\n'.join(selected)

    def search_code(self,
                    worktree_path: Path,
                    pattern: str,
                    paths: Optional[Sequence[str]] = None,
                    max_results: int = 60) -> str:
        cmd = ['rg', '-n', '--hidden', '--color', 'never', '-S', pattern]
        if paths:
            cmd.append('--')
            cmd.extend(paths)
        process = self._run(cmd, cwd=worktree_path, timeout=30, check=False)
        output = process.stdout.strip()
        if process.returncode not in (0, 1):
            raise GitCommandError(f'rg failed:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}')
        if not output:
            return 'No matches found.'
        lines = output.splitlines()[:max_results]
        if len(output.splitlines()) > max_results:
            lines.append(f'... truncated after {max_results} matches')
        return '\n'.join(lines)

    def write_patch_file(self, artifact_dir: Path, filename: str, patch_text: str) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        patch_path = artifact_dir / filename
        save_text_to_file(str(patch_path), patch_text)
        return patch_path

    def check_patch(self, worktree_path: Path, patch_text: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.patch', delete=False) as tmp:
            tmp.write(patch_text)
            tmp_path = tmp.name
        try:
            return self._run(['git', 'apply', '--check', '--recount', '--ignore-space-change',
                              '--whitespace=nowarn', '--verbose', tmp_path],
                             cwd=worktree_path,
                             timeout=120,
                             check=False)
        finally:
            os.unlink(tmp_path)

    def apply_patch(self, worktree_path: Path, patch_text: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.patch', delete=False) as tmp:
            tmp.write(patch_text)
            tmp_path = tmp.name
        try:
            return self._run(['git', 'apply', '--recount', '--ignore-space-change',
                              '--whitespace=nowarn', '--verbose', tmp_path],
                             cwd=worktree_path,
                             timeout=120,
                             check=False)
        finally:
            os.unlink(tmp_path)

    def current_changed_files(self, worktree_path: Path) -> List[str]:
        output = self.git('diff', '--name-only', cwd=worktree_path)
        return [line.strip() for line in output.splitlines() if line.strip()]

    def export_current_patch(self, worktree_path: Path) -> str:
        return self.git('diff', '--binary', '--no-ext-diff', cwd=worktree_path, timeout=120)

    def diff_vs_commit(self, worktree_path: Path, commit: str, max_chars: int = 20000) -> str:
        stat = self.git('diff', '--stat', '--no-ext-diff', commit, cwd=worktree_path, timeout=120)
        diff = self.git('diff', '--no-ext-diff', '--unified=3', commit, cwd=worktree_path, timeout=120)
        text = stat.strip()
        if diff.strip():
            text = (text + '\n\n' + truncate_middle(diff, max_chars)).strip()
        return text

    def tree_matches_commit(self, worktree_path: Path, commit: str) -> bool:
        process = self._run(['git', 'diff', '--quiet', '--no-ext-diff', commit],
                            cwd=worktree_path,
                            timeout=120,
                            check=False)
        return process.returncode == 0

    def save_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_text_to_file(str(path), json.dumps(payload, ensure_ascii=False, indent=2))
