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
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, Message
from qwen_agent.memory import Memory
from qwen_agent.tools import BaseTool

from .fncall_agent import FnCallAgent


class ArticleAgent(FnCallAgent):
    """An agent specialized for article writing and summarization tasks.

    It can read documents/URLs and produce articles based on the content.
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
        """Initialize the ArticleAgent.

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

        default_system = (
            'You are a professional article writer. '
            'You can read documents and URLs to gather information, '
            'then write high-quality articles based on the provided content. '
            'When writing, ensure the article is well-structured, informative, and engaging.'
        )
        if system_message == DEFAULT_SYSTEM_MESSAGE:
            system_message = default_system

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            files=files,
            **kwargs,
        )
