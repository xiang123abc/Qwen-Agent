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

"""Example: Parallel function calling with Qwen Agent."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import FnCallAgent


def test():
    """Test parallel function calling agent."""
    llm_cfg = {'model': 'qwen-max'}

    tools = [{
        'name': 'weather',
        'description': 'Get weather for a location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {'type': 'string'},
            },
            'required': ['location'],
        },
    }, {
        'name': 'calculator',
        'description': 'Evaluate a math expression',
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {'type': 'string'},
            },
            'required': ['expression'],
        },
    }]

    agent = FnCallAgent(llm=llm_cfg, function_list=tools)

    messages = [{'role': 'user', 'content': 'What is the weather in Beijing and what is 5 * 3?'}]

    response = agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
