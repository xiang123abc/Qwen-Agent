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

"""Example: ReAct-based data analysis with code interpreter."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import FnCallAgent
from qwen_agent.llm.schema import Message, ContentItem


def test(query='pd.head the file first and then help me draw a line chart to show the changes in stock prices',
         file='examples/resource/stock_prices.csv'):
    """Test ReAct data analysis."""
    llm_cfg = {'model': 'qwen-max'}

    agent = FnCallAgent(
        llm=llm_cfg,
        function_list=['code_interpreter'],
        description='Data analysis agent',
    )

    messages = [
        Message('user', [
            ContentItem(text=query),
            ContentItem(file=file),
        ])
    ]

    response = agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
