"""面向漏洞输入的 closed-loop Linux kernel autopatch Agent。"""

from pathlib import Path
from typing import Dict, Optional, Union

from qwen_agent.llm import BaseChatModel, get_chat_model

from qwen_agent.kernel_patch.autopatch import ClosedLoopPatchPipeline, LLMJsonPatchReasoner
from qwen_agent.kernel_patch.autopatch_models import AutoPatchResult, VulnerabilityInput
from qwen_agent.kernel_patch.compile_validator import KernelCompileValidator
from qwen_agent.kernel_patch.git_ops import KernelRepoManager


class ClosedLoopKernelPatchAgent:
    """从漏洞输入到 patch 生成/验证的闭环 Agent。"""

    def __init__(self,
                 llm: Optional[Union[Dict, BaseChatModel]],
                 repo_manager: KernelRepoManager,
                 compile_validator: Optional[KernelCompileValidator] = None):
        if llm is None:
            raise ValueError('ClosedLoopKernelPatchAgent requires an llm or model config.')
        self.llm = get_chat_model(llm) if isinstance(llm, dict) else llm
        self.repo_manager = repo_manager
        self.pipeline = ClosedLoopPatchPipeline(repo_manager,
                                                reasoner=LLMJsonPatchReasoner(self.llm),
                                                compile_validator=compile_validator)

    def run(self,
            vulnerability: VulnerabilityInput,
            worktree_path: Optional[Union[str, Path]] = None,
            revision: str = 'HEAD',
            session_name: str = 'autopatch',
            recreate_worktree: bool = True,
            max_iterations: int = 2) -> AutoPatchResult:
        """执行一次完整闭环。

        - 若传入 `worktree_path`，直接在该 worktree 上工作；
        - 否则基于 `revision` 自动创建 detached worktree。
        """
        if worktree_path is not None:
            return self.pipeline.run(vulnerability,
                                     Path(worktree_path),
                                     session_name=session_name,
                                     max_iterations=max_iterations)
        return self.pipeline.run_on_revision(vulnerability,
                                             revision=revision,
                                             session_name=session_name,
                                             recreate_worktree=recreate_worktree,
                                             max_iterations=max_iterations)
