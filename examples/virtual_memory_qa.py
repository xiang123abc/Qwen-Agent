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

"""Example: Virtual memory Q&A system."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import FnCallAgent


def test():
    """Test virtual memory Q&A."""
    llm_cfg = {'model': 'qwen-max'}

    # Agent with retrieval capability
    agent = FnCallAgent(
        llm=llm_cfg,
        function_list=['retrieval'],
        description='Q&A system with memory',
    )

    messages = [
        {'role': 'user', 'content': '根据提供给我的文档回答问题：什么是机器学习？'},
    ]

    response = agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
