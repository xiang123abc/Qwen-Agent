"""本地编译验证器。

用于在 patch 应用成功后做额外质量门禁：仅编译受影响的 `.c -> .o` 目标，
快速判断补丁是否引入明显编译错误。
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from .models import CompileValidationResult


class KernelCompileValidator:
    """面向 Linux 内核 worktree 的轻量编译校验。"""

    def __init__(self, repo_root: str, jobs: int = 4, timeout_sec: int = 300):
        """初始化校验器参数。"""
        self.repo_root = Path(repo_root).resolve()
        self.jobs = jobs
        self.timeout_sec = timeout_sec

    def _find_config(self, worktree_path: Path) -> Optional[Path]:
        """查找可用 `.config`，优先使用 worktree 本地配置。"""
        local = worktree_path / '.config'
        if local.exists():
            return local
        shared = self.repo_root / '.config'
        if shared.exists():
            return shared
        return None

    def validate(self, worktree_path: Path, changed_files: List[str]) -> CompileValidationResult:
        """执行最小化目标编译校验并返回结构化结果。"""
        config = self._find_config(worktree_path)
        if config is None:
            return CompileValidationResult(status='skipped', output='No .config found; skipping compile validation.')

        # 仅对变更过的 C 源文件尝试构建对应目标，避免全量编译开销。
        targets = []
        for path in changed_files:
            if path.endswith('.c'):
                targets.append(path[:-2] + '.o')
        if not targets:
            return CompileValidationResult(status='skipped', output='No .c targets detected for compile validation.')

        cmd = ['make', '-C', str(worktree_path), f'-j{self.jobs}', *targets]
        env = None
        if config.parent != worktree_path:
            env = {'KCONFIG_CONFIG': str(config)}
        process = subprocess.run(cmd,
                                 cwd=str(worktree_path),
                                 capture_output=True,
                                 text=True,
                                 encoding='utf-8',
                                 errors='replace',
                                 timeout=self.timeout_sec,
                                 env=env)
        output = (process.stdout + '\n' + process.stderr).strip()
        status = 'passed' if process.returncode == 0 else 'failed'
        return CompileValidationResult(status=status,
                                       command=' '.join(cmd),
                                       output=output[:12000],
                                       validated_targets=targets)
