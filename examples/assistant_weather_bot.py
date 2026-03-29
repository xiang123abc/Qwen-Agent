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

"""Example: Assistant weather bot with file reading support."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import Assistant


def test(query='海淀区天气', file=None):
    """Test weather bot with optional file attachment."""
    llm_cfg = {'model': 'qwen-max'}

    # System message for weather assistant
    system = '你扮演一个天气预报助手，你可以查询天气信息。'

    # Use amap_weather tool
    tools = ['amap_weather']
    agent = Assistant(llm=llm_cfg, system_message=system, function_list=tools)

    messages = [{'role': 'user', 'content': query}]
    if file:
        messages[0]['file'] = file

    response = agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
