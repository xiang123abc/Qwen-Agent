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

"""Example: Group chat chess game."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import FnCallAgent


def test(query='开始吧'):
    """Test chess game in group chat."""
    llm_cfg = {'model': 'qwen-max'}

    chess_agent = FnCallAgent(
        llm=llm_cfg,
        name='ChessMaster',
        description='A chess playing agent',
        system_message='你是一个国际象棋大师，可以和用户下棋。',
    )

    messages = [{'role': 'user', 'content': query}]

    response = chess_agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
