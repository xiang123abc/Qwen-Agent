from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

import json5

from qwen_agent import Agent
from qwen_agent.llm.schema import ASSISTANT, USER, Message

from .config import KernelPatchConfig
from .pipeline import KernelPatchPipeline


class KernelPatchAgent(Agent):

    def __init__(self, *args, debug_confirm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_confirm = debug_confirm

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        user_message = next((message for message in reversed(messages) if message.role == USER), None)
        if user_message is None or not isinstance(user_message.content, str):
            raise ValueError('KernelPatchAgent expects the latest user message to be a JSON task payload or config path')

        payload = user_message.content.strip()
        if payload.endswith('.json') and Path(payload).exists():
            config_data = json5.loads(Path(payload).read_text(encoding='utf-8'))
        else:
            config_data = json5.loads(payload)

        config = KernelPatchConfig.model_validate(config_data)
        pipeline_llm = self.llm
        if pipeline_llm is None and config.llm:
            pipeline_llm = config.llm.to_qwen_cfg()
        pipeline = KernelPatchPipeline(llm=pipeline_llm, debug_confirm=self.debug_confirm)
        result = pipeline.run(config)
        response = {
            'cve_id': result.cve_id,
            'success': result.success,
            'summary': result.summary,
            'analysis_path': result.analysis_path,
            'patch_path': result.patch_path,
            'trace_path': result.trace_path,
        }
        yield [Message(role=ASSISTANT, content=json.dumps(response, ensure_ascii=False, indent=2))]
