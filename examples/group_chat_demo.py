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

"""Example: Multi-agent group chat demo."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import FnCallAgent


def test():
    """Test group chat demo."""
    # Create multiple agents
    llm_cfg = {'model': 'qwen-max'}

    assistant1 = FnCallAgent(llm=llm_cfg, name='Alice', description='A helpful assistant')
    assistant2 = FnCallAgent(llm=llm_cfg, name='Bob', description='A creative assistant')

    messages = [{'role': 'user', 'content': '你们两个分别介绍一下自己'}]

    # Run both agents
    response1 = assistant1.run(messages)
    response2 = assistant2.run(messages)

    print("Alice:")
    for rsp in response1:
        print(rsp)

    print("\nBob:")
    for rsp in response2:
        print(rsp)


if __name__ == '__main__':
    test()
