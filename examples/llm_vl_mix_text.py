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

"""Example: Vision language model with mixed text and images."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ContentItem


def test():
    """Test VL mixed text/image."""
    llm_cfg = {'model': 'qwen-vl-max'}

    agent = Assistant(llm=llm_cfg)

    messages = [
        Message('user', [
            ContentItem(text='描述一下这张图片'),
            ContentItem(image='https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg'),
        ])
    ]

    response = agent.run(messages)
    for rsp in response:
        print(rsp)


if __name__ == '__main__':
    test()
