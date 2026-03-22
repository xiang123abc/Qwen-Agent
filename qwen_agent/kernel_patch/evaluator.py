"""Patch 评估器。

职责：
1. 校验模型输出 patch 是否可解析、可应用；
2. 将应用后的工作树与社区修复提交对比；
3. 生成可用于重试的结构化反馈；
4. 可选执行编译验证。
"""

from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import List, Optional

from .compile_validator import KernelCompileValidator
from .git_ops import KernelRepoManager
from .models import KernelCaseBundle, PatchEvaluation
from .ordering_constraints import detect_ordering_constraints


def _normalize_patch(text: str) -> str:
    """归一化 patch 文本，便于做相似度比较。"""
    lines = []
    for line in text.splitlines():
        if line.startswith('index '):
            continue
        if line.startswith('@@'):
            lines.append('@@')
            continue
        lines.append(line.rstrip())
    return '\n'.join(lines).strip()


HUNK_HEADER_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def _infer_hunk_header(manager: KernelRepoManager, worktree_path: Path, relative_path: str, hunk_body: List[str]) -> str:
    """根据 hunk 正文反推合法 `@@ -a,b +c,d @@` 头部。"""
    source_path = worktree_path / relative_path
    if not source_path.exists():
        return ''

    file_lines = source_path.read_text(encoding='utf-8', errors='replace').splitlines()
    original_lines = []
    new_count = 0
    for line in hunk_body:
        if not line:
            original_lines.append('')
            new_count += 1
            continue
        if line.startswith('\\'):
            continue
        if line[0] in (' ', '-'):
            original_lines.append(line[1:])
        if line[0] in (' ', '+'):
            new_count += 1

    if not original_lines:
        return ''

    old_count = len(original_lines)
    # 在当前父树文件里匹配“旧内容窗口”，命中后即可反推 old_start。
    for idx in range(0, len(file_lines) - old_count + 1):
        if file_lines[idx:idx + old_count] == original_lines:
            old_start = idx + 1
            old_count_str = str(old_count)
            new_count_str = str(new_count)
            return f'@@ -{old_start},{old_count_str} +{old_start},{new_count_str} @@'
    return ''


def _repair_patch_headers(manager: KernelRepoManager, worktree_path: Path, patch_text: str) -> str:
    """修复模型产出中格式异常的 hunk header。"""
    lines = patch_text.splitlines()
    repaired = []
    current_path = ''
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            parts = line.split()
            if len(parts) >= 4:
                current_path = parts[2][2:] if parts[2].startswith('a/') else parts[2]
            repaired.append(line)
            i += 1
            continue

        if line.startswith('--- ') or line.startswith('+++ '):
            repaired.append(line)
            i += 1
            continue

        # 遇到非法 hunk 头时，尝试用正文内容推断替换。
        if line.startswith('@@') and not HUNK_HEADER_RE.match(line):
            j = i + 1
            hunk_body = []
            while j < len(lines) and not lines[j].startswith('diff --git ') and not lines[j].startswith('@@'):
                hunk_body.append(lines[j])
                j += 1
            inferred = _infer_hunk_header(manager, worktree_path, current_path, hunk_body)
            repaired.append(inferred or line)
            repaired.extend(hunk_body)
            i = j
            continue

        repaired.append(line)
        i += 1

    return '\n'.join(repaired).rstrip() + '\n'


def _summarize_tree_mismatch(diff_text: str, max_lines: int = 4) -> str:
    """从剩余 diff 提炼“保留/避免”提示，指导下一轮生成。"""
    restore_lines = []
    avoid_lines = []
    for line in diff_text.splitlines():
        if line.startswith(('diff --git', 'index ', '--- ', '+++ ', '@@')):
            continue
        if line.startswith('-') and not line.startswith('--- '):
            content = line[1:].strip()
            if content:
                restore_lines.append(content)
        elif line.startswith('+') and not line.startswith('+++ '):
            content = line[1:].strip()
            if content:
                avoid_lines.append(content)

    hints = []
    if restore_lines:
        hints.append('Keep or restore these target lines exactly:')
        hints.extend(f'- {line}' for line in restore_lines[:max_lines])
    if avoid_lines:
        hints.append('Do not leave these attempt-only lines in the final patch:')
        hints.extend(f'- {line}' for line in avoid_lines[:max_lines])
    if not hints:
        return 'Your patch still differs from the community fix. Preserve statement order and make only the minimal edit.'
    hints.append('Do not reorder existing declarations or statements unless the community patch does so.')
    return '\n'.join(hints)


class PatchEvaluator:
    """执行 patch 应用与对比评估的主类。"""

    def __init__(self,
                 manager: KernelRepoManager,
                 feedback_style: str = 'summary',
                 compile_validator: Optional[KernelCompileValidator] = None):
        """初始化评估器依赖与反馈风格。"""
        self.manager = manager
        self.feedback_style = feedback_style
        self.compile_validator = compile_validator

    def evaluate(self,
                 bundle: KernelCaseBundle,
                 worktree_path: Path,
                 patch_text: str,
                 attempt_dir: Path) -> PatchEvaluation:
        """评估单次 patch 尝试并返回结构化结果。"""
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if not patch_text or not patch_text.strip():
            return PatchEvaluation(patch_found=False,
                                   patch_apply_ok=False,
                                   tree_match=False,
                                   similarity=0.0,
                                   error_message='Model did not return a patch.',
                                   feedback_message='Your last answer did not contain a valid unified diff patch block.',
                                   categories=['missing_patch'])

        # 先做 patch 头修复，再进入 apply check 流程。
        patch_text = _repair_patch_headers(self.manager, worktree_path, patch_text)
        patch_path = self.manager.write_patch_file(attempt_dir, 'candidate.patch', patch_text)
        check_result = self.manager.check_patch(worktree_path, patch_text)
        if check_result.returncode != 0:
            message = (check_result.stdout + '\n' + check_result.stderr).strip()
            return PatchEvaluation(patch_found=True,
                                   patch_apply_ok=False,
                                   tree_match=False,
                                   similarity=0.0,
                                   error_message=message,
                                   feedback_message='Your patch did not apply cleanly with git apply. '
                                   f'Fix the patch format and surrounding context.\n\n{message}',
                                   categories=['patch_apply_failed'],
                                   generated_patch_path=patch_path)

        apply_result = self.manager.apply_patch(worktree_path, patch_text)
        if apply_result.returncode != 0:
            message = (apply_result.stdout + '\n' + apply_result.stderr).strip()
            return PatchEvaluation(patch_found=True,
                                   patch_apply_ok=False,
                                   tree_match=False,
                                   similarity=0.0,
                                   error_message=message,
                                   feedback_message='git apply --check passed but apply still failed. '
                                   f'Inspect the error and regenerate the patch.\n\n{message}',
                                   categories=['patch_apply_failed'],
                                   generated_patch_path=patch_path)

        # patch 已成功应用，开始收集结果并对齐社区修复树。
        touched_files = self.manager.current_changed_files(worktree_path)
        generated_patch = self.manager.export_current_patch(worktree_path)
        generated_patch_path = self.manager.write_patch_file(attempt_dir, 'applied.patch', generated_patch)
        similarity = SequenceMatcher(None,
                                     _normalize_patch(generated_patch),
                                      _normalize_patch(bundle.community_patch)).ratio()
        tree_match = self.manager.tree_matches_commit(worktree_path, bundle.fix_commit)
        remaining_diff = ''
        feedback = 'Patch applied successfully and matches the community-fixed tree.'
        categories: List[str] = ['success']
        compile_validation = None
        if self.compile_validator is not None:
            compile_validation = self.compile_validator.validate(worktree_path, touched_files)
            if compile_validation.status == 'failed':
                categories = ['compile_failed']
                feedback = ('Patch applies as text, but local compile validation failed.\n\n'
                            f'Command: {compile_validation.command}\n'
                            f'Output:\n{compile_validation.output}')
        if not tree_match:
            # 树不一致时给出可操作反馈（可选完整 diff 或摘要）。
            remaining_diff = self.manager.diff_vs_commit(worktree_path, bundle.fix_commit)
            categories = ['tree_mismatch']
            if set(touched_files) != set(bundle.changed_files):
                categories.append('wrong_files')
            categories.extend(category for category in detect_ordering_constraints(remaining_diff) if category not in categories)
            if self.feedback_style == 'full_diff':
                feedback = ('Your patch applies, but the resulting tree still differs from the community fix. '
                            'Regenerate the patch while staying minimal.\n\n'
                            f'{remaining_diff}')
            else:
                mismatch_summary = _summarize_tree_mismatch(remaining_diff)
                feedback = ('Your patch applies, but the resulting tree still differs from the community fix. '
                            'Regenerate the patch from the original parent tree with a closer textual match.\n\n'
                            f'Changed files now: {touched_files}\n'
                            f'Community files: {bundle.changed_files}\n'
                            f'Guidance:\n{mismatch_summary}')

        return PatchEvaluation(patch_found=True,
                               patch_apply_ok=True,
                               tree_match=tree_match,
                               similarity=similarity,
                               touched_files=touched_files,
                               remaining_diff=remaining_diff,
                               feedback_message=feedback,
                               categories=categories,
                               generated_patch_path=generated_patch_path,
                               compile_validation=compile_validation)
