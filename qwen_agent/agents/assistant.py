# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.llm import BaseChatModel, get_chat_model
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.memory import Memory
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import extract_files_from_messages

from .fncall_agent import FnCallAgent


class Assistant(FnCallAgent):
    """A general-purpose assistant agent that can use tools and handle various file types.

    This is the primary user-facing agent, similar to ChatGPT's Assistant API.
    It supports function calling, file handling, vision, and long context.
    """

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        name: Optional[str] = None,
        description: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize the Assistant agent.

        Args:
            function_list: List of tool names, configurations, or Tool objects.
            llm: LLM model configuration or model object.
            system_message: System message for the agent.
            name: Agent name.
            description: Agent description for multi-agent routing.
            files: Initial file URLs for the agent.
        """
        if isinstance(llm, dict):
            llm = get_chat_model(llm)

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            files=files,
            **kwargs,
        )

        # Assistant-specific file handling
        self._files = files or []

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        """Call a tool with file support."""
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exist.'

        tool = self.function_map[tool_name]
        if tool.file_access:
            files = []
            if 'messages' in kwargs:
                files = extract_files_from_messages(kwargs['messages'], include_images=True)
            files = files + self.mem.system_files if hasattr(self, 'mem') else files
            return super()._call_tool(tool_name, tool_args, files=files, **kwargs)
        return super()._call_tool(tool_name, tool_args, **kwargs)
