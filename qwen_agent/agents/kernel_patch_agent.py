from pathlib import Path
from typing import Dict, Optional, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message

from qwen_agent.kernel_patch.git_ops import KernelRepoManager
from qwen_agent.kernel_patch.models import KernelCaseBundle, PatchCandidate, extract_patch_from_response, strip_patch_from_response
from qwen_agent.kernel_patch.prompts import build_analysis_prompt, build_patch_prompt, render_system_prompt
from qwen_agent.kernel_patch.prompt_tuner import PromptProfile
from qwen_agent.kernel_patch.tools import build_kernel_tools


def _extract_text(message_list) -> str:
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

    def __init__(self,
                 llm: Optional[Union[Dict, BaseChatModel]],
                 repo_manager: KernelRepoManager,
                 prompt_profile: Optional[PromptProfile] = None):
        self.llm = llm
        self.repo_manager = repo_manager
        self.prompt_profile = prompt_profile or PromptProfile()

    def _build_bot(self, bundle: KernelCaseBundle, worktree_path: Path) -> FnCallAgent:
        tools = build_kernel_tools(self.repo_manager, bundle, worktree_path)
        return FnCallAgent(llm=self.llm,
                           system_message=render_system_prompt(self.prompt_profile.render_rules()),
                           function_list=tools,
                           name='kernel_patch_agent')

    def _run_single_prompt(self, bot: FnCallAgent, prompt: str) -> str:
        messages = [Message(role='user', content=prompt)]
        last = None
        for last in bot.run(messages=messages, lang='en'):
            continue
        return _extract_text(last or [])

    def generate_candidate(self,
                           bundle: KernelCaseBundle,
                           worktree_path: Path,
                           feedback_text: str = '') -> PatchCandidate:
        bot = self._build_bot(bundle, worktree_path)
        current_context_excerpt = self.repo_manager.current_hunk_context(bundle, worktree_path)
        analysis_text = self._run_single_prompt(bot, build_analysis_prompt(bundle, current_context_excerpt))
        patch_prompt = build_patch_prompt(analysis_text,
                                          current_context_excerpt=current_context_excerpt,
                                          feedback_text=feedback_text)
        raw_response = self._run_single_prompt(bot, patch_prompt)
        patch_text = extract_patch_from_response(raw_response)
        analysis_text = analysis_text.strip()
        if patch_text:
            raw_analysis = strip_patch_from_response(raw_response, patch_text)
            if raw_analysis:
                analysis_text = (analysis_text + '\n\n' + raw_analysis).strip()
        return PatchCandidate(analysis_text=analysis_text, patch_text=patch_text, raw_response=raw_response)
