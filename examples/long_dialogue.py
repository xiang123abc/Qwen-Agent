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

"""Example: Long dialogue handling with memory."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import Assistant


def test():
    """Test long dialogue with memory."""
    llm_cfg = {'model': 'qwen-max'}

    agent = Assistant(
        llm=llm_cfg,
        description='Assistant with long dialogue support',
    )

    # Simulate a long conversation
    messages = [
        {'role': 'user', 'content': '你好，我叫张三，我是一名软件工程师。'},
        {'role': 'assistant', 'content': '你好张三！很高兴认识你。作为软件工程师，你主要使用什么编程语言？'},
        {'role': 'user', 'content': '我主要使用Python和JavaScript。'},
        {'role': 'assistant', 'content': '很好！Python和JavaScript都是非常流行的语言。'},
        {'role': 'user', 'content': '你能记住我的名字和职业吗？'},
    ]

    response = agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
