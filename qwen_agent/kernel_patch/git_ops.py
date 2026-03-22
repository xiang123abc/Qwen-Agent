"""Kernel 仓库操作封装层。

该模块统一处理：
- git 元数据提取（commit、diff、show）；
- worktree 创建/重置/清理；
- 代码片段读取、符号检索、块级上下文定位；
- patch 校验/应用与评估产物落盘。
"""

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

from .block_index import build_file_block_index, find_block_by_name, locate_block_by_line, nearest_blocks
from .models import CodeBlock, FileBlockIndex, KernelCaseBundle, PatchCase, truncate_middle


class GitCommandError(RuntimeError):
    """git 命令执行失败异常。"""
    pass


def _ensure_within(root: Path, candidate: Path) -> None:
    """校验路径必须位于 root 内，防止路径穿越。"""
    root = root.resolve()
    candidate = candidate.resolve()
    if os.path.commonpath([str(root), str(candidate)]) != str(root):
        raise ValueError(f'Path `{candidate}` escapes repository root `{root}`')


class KernelRepoManager:
    """kernel patch 任务的仓库管理器。"""

    def __init__(self, repo_root: str, workspace_root: str):
        """初始化仓库与工作区目录结构。"""
        self.repo_root = Path(repo_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.worktree_root = self.workspace_root / 'worktrees'
        self.artifacts_root = self.workspace_root / 'artifacts'
        self._block_index_cache: dict[tuple[str, str], FileBlockIndex] = {}
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        if not (self.repo_root / '.git').exists():
            raise ValueError(f'Not a git repository: {self.repo_root}')

    def _run(self,
             cmd: Sequence[str],
             cwd: Optional[Path] = None,
             timeout: int = 60,
             check: bool = True) -> subprocess.CompletedProcess:
        """执行外部命令并按需抛出结构化错误。"""
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
        """执行 git 子命令并返回 stdout。"""
        return self._run(['git', *args], cwd=cwd, timeout=timeout).stdout

    def rev_parse(self, revision: str) -> str:
        """解析任意 revision 到完整 commit id。"""
        return self.git('rev-parse', revision).strip()

    def parent_commit(self, commit: str) -> str:
        """获取 commit 的父提交。"""
        return self.rev_parse(f'{commit}^')

    def changed_files(self, commit: str) -> List[str]:
        """列出某个提交直接修改的文件。"""
        output = self.git('diff-tree', '--no-commit-id', '--name-only', '-r', commit)
        return [line.strip() for line in output.splitlines() if line.strip()]

    def commit_subject(self, commit: str) -> str:
        """读取提交标题（subject）。"""
        return self.git('log', '--format=%s', '-n', '1', commit).strip()

    def commit_message(self, commit: str) -> str:
        """读取完整提交消息（body）。"""
        return self.git('log', '--format=%B', '-n', '1', commit).strip()

    def diff_stat(self, base_commit: str, fix_commit: str) -> str:
        """统计 base 与 fix 之间的文件变更概览。"""
        return self.git('diff', '--stat', base_commit, fix_commit).strip()

    def reference_show(self, fix_commit: str, paths: Optional[Iterable[str]] = None) -> str:
        """获取社区修复提交的 `git show -W` 视图。"""
        cmd = ['show', '--patch', '--no-ext-diff', '--format=fuller', '--stat', '-W', fix_commit]
        if paths:
            cmd.append('--')
            cmd.extend(paths)
        return self.git(*cmd, timeout=120)

    def community_patch(self, base_commit: str, fix_commit: str) -> str:
        """导出社区修复的标准 unified diff。"""
        return self.git('diff', '--binary', '--no-ext-diff', base_commit, fix_commit, timeout=120)

    def prepare_case_bundle(self, case: PatchCase, reference_char_limit: int = 50000) -> KernelCaseBundle:
        """构建单个 case 的完整运行上下文（KernelCaseBundle）。"""
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
        """搜索符号并返回命中附近的扩展源码上下文。"""
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
        """按社区 patch 的 hunk 位置，提取父树当前代码片段。"""
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

            # 每个文件限制提取 hunk 数，控制 prompt 体积。
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

    def get_block_index(self, worktree_path: Path, relative_path: str) -> FileBlockIndex:
        """读取并缓存文件块索引。"""
        key = (str(worktree_path.resolve()), relative_path)
        if key not in self._block_index_cache:
            full_path = (worktree_path / relative_path).resolve()
            _ensure_within(worktree_path, full_path)
            text = read_text_from_file(str(full_path))
            self._block_index_cache[key] = build_file_block_index(relative_path, text)
        return self._block_index_cache[key]

    def locate_block(self, worktree_path: Path, relative_path: str, line_no: int) -> Optional[CodeBlock]:
        """按行号定位所属代码块。"""
        return locate_block_by_line(self.get_block_index(worktree_path, relative_path), line_no)

    def nearest_blocks(self, worktree_path: Path, relative_path: str, line_no: int):
        """返回目标行前后最近代码块。"""
        return nearest_blocks(self.get_block_index(worktree_path, relative_path), line_no)

    def read_block(self,
                   worktree_path: Path,
                   relative_path: str,
                   symbol: Optional[str] = None,
                   kind: Optional[str] = None,
                   line_no: Optional[int] = None) -> str:
        """读取顶层代码块文本（按 symbol 或 line_no 定位）。"""
        index = self.get_block_index(worktree_path, relative_path)
        block = None
        if symbol:
            kinds = (kind,) if kind else None
            block = find_block_by_name(index, symbol, kinds=kinds)
        if block is None and line_no is not None:
            block = locate_block_by_line(index, line_no)
        if block is None:
            return f'No block found in {relative_path}.'
        return self.read_file_slice(worktree_path, relative_path, block.start_line, block.end_line)

    def read_insertion_context(self,
                               worktree_path: Path,
                               relative_path: str,
                               anchor_before: str = '',
                               anchor_after: str = '',
                               context_lines: int = 12) -> str:
        """读取新增块插入位置附近的上下文。"""
        index = self.get_block_index(worktree_path, relative_path)
        before_block = find_block_by_name(index, anchor_before) if anchor_before else None
        after_block = find_block_by_name(index, anchor_after) if anchor_after else None
        if before_block:
            start_line = max(1, before_block.end_line - context_lines)
            end_line = before_block.end_line + context_lines
            return self.read_file_slice(worktree_path, relative_path, start_line, end_line)
        if after_block:
            start_line = max(1, after_block.start_line - context_lines)
            end_line = after_block.start_line + context_lines
            return self.read_file_slice(worktree_path, relative_path, start_line, end_line)
        return self.read_file_slice(worktree_path, relative_path, 1, min(context_lines * 2, 80))

    def find_type_definition(self,
                             worktree_path: Path,
                             type_name: str,
                             paths: Optional[Sequence[str]] = None,
                             max_matches: int = 3) -> str:
        """查找并返回类型定义（struct/union/enum/typedef）。"""
        normalized = type_name.replace('struct ', '').replace('union ', '').replace('enum ', '').strip()
        candidate_paths = list(paths or [])
        if not candidate_paths:
            pattern = rf'\b(struct|union|enum|typedef)\s+{re.escape(normalized)}\b'
            hits = self.search_code(worktree_path, pattern=pattern, paths=None, max_results=max_matches)
            candidate_paths = []
            for line in hits.splitlines():
                parts = line.split(':', 2)
                if len(parts) >= 2 and parts[0] not in candidate_paths:
                    candidate_paths.append(parts[0])
        matches = []
        for path in candidate_paths[:max_matches]:
            index = self.get_block_index(worktree_path, path)
            for block in index.blocks:
                if block.kind in {'struct', 'union', 'enum', 'typedef'} and block.name == normalized:
                    snippet = self.read_file_slice(worktree_path, path, block.start_line, block.end_line)
                    matches.append(f'## {path}:{block.start_line}\n{snippet}')
                    break
        return '\n\n'.join(matches) if matches else f'No type definition found for {type_name}.'

    def find_macro_definition(self,
                              worktree_path: Path,
                              macro_name: str,
                              paths: Optional[Sequence[str]] = None,
                              max_matches: int = 3) -> str:
        """查找并返回宏定义块。"""
        candidate_paths = list(paths or [])
        if not candidate_paths:
            hits = self.search_code(worktree_path,
                                    pattern=rf'^\s*#define\s+{re.escape(macro_name)}\b',
                                    paths=None,
                                    max_results=max_matches)
            candidate_paths = []
            for line in hits.splitlines():
                parts = line.split(':', 2)
                if len(parts) >= 2 and parts[0] not in candidate_paths:
                    candidate_paths.append(parts[0])
        matches = []
        for path in candidate_paths[:max_matches]:
            index = self.get_block_index(worktree_path, path)
            for block in index.blocks:
                if block.kind == 'macro' and block.name == macro_name:
                    snippet = self.read_file_slice(worktree_path, path, block.start_line, block.end_line)
                    matches.append(f'## {path}:{block.start_line}\n{snippet}')
                    break
        return '\n\n'.join(matches) if matches else f'No macro definition found for {macro_name}.'

    def read_include_context(self, worktree_path: Path, relative_path: str) -> str:
        """读取文件头部 include 区段，辅助判断依赖关系。"""
        full_path = (worktree_path / relative_path).resolve()
        _ensure_within(worktree_path, full_path)
        lines = read_text_from_file(str(full_path)).splitlines()
        selected = []
        for idx, line in enumerate(lines, start=1):
            if line.startswith('#include') or (selected and not line.strip()):
                selected.append(f'{idx:6d} {line}')
                continue
            if selected:
                break
        return '\n'.join(selected) if selected else f'No include context found for {relative_path}.'

    def worktree_path(self, bundle: KernelCaseBundle) -> Path:
        """根据 case slug 计算工作树路径。"""
        return self.worktree_root / bundle.case.slug

    def remove_worktree(self, worktree_path: Path) -> None:
        """安全移除工作树目录并清理 git worktree 元数据。"""
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
        """创建（或复用）指向 base_commit 的 detached worktree。"""
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
        """将 worktree 强制重置到指定 base commit。"""
        if not worktree_path.exists():
            raise ValueError(f'Worktree does not exist: {worktree_path}')
        self.git('checkout', '--detach', base_commit, cwd=worktree_path, timeout=120)
        self.git('reset', '--hard', base_commit, cwd=worktree_path, timeout=120)
        self.git('clean', '-fdq', cwd=worktree_path, timeout=120)

    def read_file_slice(self, worktree_path: Path, relative_path: str, start_line: int, end_line: int) -> str:
        """按行号范围读取文件切片，并附带行号前缀。"""
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
        """在 worktree 中搜索代码，优先使用 `rg`。"""
        if shutil.which('rg'):
            cmd = ['rg', '-n', '--hidden', '--color', 'never', '-S', pattern]
            if paths:
                cmd.append('--')
                cmd.extend(paths)
            process = self._run(cmd, cwd=worktree_path, timeout=30, check=False)
            backend = 'rg'
        elif shutil.which('git'):
            cmd = ['git', 'grep', '-n', '--no-color', '-E', pattern]
            if paths:
                cmd.append('--')
                cmd.extend(paths)
            process = self._run(cmd, cwd=worktree_path, timeout=30, check=False)
            backend = 'git grep'
        else:
            cmd = ['grep', '-RIn', '-E', pattern]
            if paths:
                cmd.extend(paths)
            else:
                cmd.append('.')
            process = self._run(cmd, cwd=worktree_path, timeout=30, check=False)
            backend = 'grep'

        output = process.stdout.strip()
        # 0: 有结果；1: 无结果；其他返回码视为执行异常。
        if process.returncode not in (0, 1):
            raise GitCommandError(f'{backend} failed:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}')
        if not output:
            return 'No matches found.'
        lines = output.splitlines()[:max_results]
        if len(output.splitlines()) > max_results:
            lines.append(f'... truncated after {max_results} matches')
        return '\n'.join(lines)

    def write_patch_file(self, artifact_dir: Path, filename: str, patch_text: str) -> Path:
        """把 patch 文本写入 artifact 目录。"""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        patch_path = artifact_dir / filename
        save_text_to_file(str(patch_path), patch_text)
        return patch_path

    def check_patch(self, worktree_path: Path, patch_text: str) -> subprocess.CompletedProcess:
        """执行 `git apply --check` 进行补丁预检查。"""
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
        """执行 `git apply` 将补丁应用到当前 worktree。"""
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
        """返回当前 worktree 未提交变更文件列表。"""
        output = self.git('diff', '--name-only', cwd=worktree_path)
        return [line.strip() for line in output.splitlines() if line.strip()]

    def export_current_patch(self, worktree_path: Path) -> str:
        """导出当前 worktree 相对 HEAD 的 patch。"""
        return self.git('diff', '--binary', '--no-ext-diff', cwd=worktree_path, timeout=120)

    def diff_vs_commit(self, worktree_path: Path, commit: str, max_chars: int = 20000) -> str:
        """对比当前树与指定 commit，并返回受限长度 diff。"""
        stat = self.git('diff', '--stat', '--no-ext-diff', commit, cwd=worktree_path, timeout=120)
        diff = self.git('diff', '--no-ext-diff', '--unified=3', commit, cwd=worktree_path, timeout=120)
        text = stat.strip()
        if diff.strip():
            text = (text + '\n\n' + truncate_middle(diff, max_chars)).strip()
        return text

    def tree_matches_commit(self, worktree_path: Path, commit: str) -> bool:
        """判断当前工作树是否与指定 commit 完全一致。"""
        process = self._run(['git', 'diff', '--quiet', '--no-ext-diff', commit],
                            cwd=worktree_path,
                            timeout=120,
                            check=False)
        return process.returncode == 0

    def save_json(self, path: Path, payload: dict) -> None:
        """保存 JSON 产物（UTF-8，缩进格式化）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_text_to_file(str(path), json.dumps(payload, ensure_ascii=False, indent=2))
