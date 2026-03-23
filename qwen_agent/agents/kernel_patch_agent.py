"""KernelPatchAgent 主流程。

该 Agent 负责把一个 CVE 修复样例（`KernelCaseBundle`）转成两阶段推理流程：
1. 先做 root-cause 分析；
2. 再输出统一 diff patch；
并将结果封装为 `PatchCandidate`。
"""

from pathlib import Path
from typing import Dict, Optional, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message

from qwen_agent.kernel_patch.case_classifier import classify_case
from qwen_agent.kernel_patch.edit_units import build_prefetched_context, parse_edit_units, summarize_edit_units
from qwen_agent.kernel_patch.git_ops import KernelRepoManager
from qwen_agent.kernel_patch.patch_alignment import build_grounded_context, refine_edit_units
from qwen_agent.kernel_patch.models import KernelCaseBundle, PatchCandidate, extract_patch_from_response, strip_patch_from_response
from qwen_agent.kernel_patch.prompts import build_analysis_prompt, build_patch_prompt, render_system_prompt
from qwen_agent.kernel_patch.prompt_tuner import PromptProfile
from qwen_agent.kernel_patch.tools import build_kernel_tools


def _extract_text(message_list) -> str:
    """从模型返回的消息列表中提取最终文本内容。

    `FnCallAgent.run` 的返回既可能是 dict，也可能是 schema 对象；
    `content` 既可能是纯字符串，也可能是多段结构化内容列表。
    """
    if not message_list:
        return ''
    last = message_list[-1]
    content = last.get('content') if isinstance(last, dict) else last.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('text'):
                    parts.append(item['text'])
            elif getattr(item, 'text', None):
                parts.append(item.text)
        return '\n'.join(parts).strip()
    return str(content)


class KernelPatchAgent:
    """面向 Linux kernel patch 任务的两阶段 Agent。"""

    def __init__(self,
                 llm: Optional[Union[Dict, BaseChatModel]],
                 repo_manager: KernelRepoManager,
                 prompt_profile: Optional[PromptProfile] = None):
        """初始化 Agent 依赖。

        - `llm`: 底层聊天模型配置或实例。
        - `repo_manager`: 负责 git/worktree 与代码检索操作。
        - `prompt_profile`: 动态提示词调优配置，未传入时使用默认配置。
        """
        self.llm = llm
        self.repo_manager = repo_manager
        self.prompt_profile = prompt_profile or PromptProfile()

    def _build_bot(self, bundle: KernelCaseBundle, worktree_path: Path) -> FnCallAgent:
        """创建带 kernel 专用工具集的函数调用 Agent。"""
        tools = build_kernel_tools(self.repo_manager, bundle, worktree_path)
        return FnCallAgent(llm=self.llm,
                           system_message=render_system_prompt(self.prompt_profile.render_rules()),
                           function_list=tools,
                           name='kernel_patch_agent')

    def _run_single_prompt(self, bot: FnCallAgent, prompt: str) -> str:
        """执行单轮 prompt 并返回最终文本。"""
        messages = [Message(role='user', content=prompt)]
        last = None
        for last in bot.run(messages=messages, lang='en'):
            continue
        return _extract_text(last or [])

    def generate_candidate(self,
                           bundle: KernelCaseBundle,
                           worktree_path: Path,
                           feedback_text: str = '') -> PatchCandidate:
        """生成一次补丁候选结果。

        执行顺序：
        1. 解析 edit unit 并做 case 分类；
        2. 生成分析 prompt，得到分析文本；
        3. 生成 patch prompt，得到原始模型响应；
        4. 从响应中提取 patch，并合并剩余分析信息。
        """
        bot = self._build_bot(bundle, worktree_path)
        edit_units = parse_edit_units(bundle, self.repo_manager, worktree_path)
        edit_units = refine_edit_units(bundle, self.repo_manager, worktree_path, edit_units)
        classification = classify_case(bundle, edit_units)
        edit_unit_summary = summarize_edit_units(edit_units)
        prefetched_context = build_prefetched_context(bundle, self.repo_manager, worktree_path, edit_units)
        grounded_context = build_grounded_context(self.repo_manager, worktree_path, edit_units)
        current_context_excerpt = (grounded_context + '\n\n' + prefetched_context).strip()
        analysis_text = self._run_single_prompt(
            bot,
            build_analysis_prompt(bundle, current_context_excerpt, classification.summary, edit_unit_summary),
        )
        patch_prompt = build_patch_prompt(analysis_text,
                                          current_context_excerpt=current_context_excerpt,
                                          case_classification=classification.summary,
                                          edit_unit_summary=edit_unit_summary,
                                          feedback_text=feedback_text)
        raw_response = self._run_single_prompt(bot, patch_prompt)
        patch_text = extract_patch_from_response(raw_response)
        analysis_text = analysis_text.strip()
        if patch_text:
            raw_analysis = strip_patch_from_response(raw_response, patch_text)
            if raw_analysis:
                analysis_text = (analysis_text + '\n\n' + raw_analysis).strip()
        return PatchCandidate(analysis_text=analysis_text, patch_text=patch_text, raw_response=raw_response)
