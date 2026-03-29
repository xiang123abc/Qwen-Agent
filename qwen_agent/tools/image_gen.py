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

from typing import Dict, Optional, Union

try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('image_gen')
class ImageGen(BaseTool):
    """Generate images from text prompts using DashScope API."""

    description = 'Generate images from text descriptions. Input is a prompt describing the desired image.'
    parameters = {
        'type': 'object',
        'properties': {
            'prompt': {
                'description': 'The text description of the image to generate',
                'type': 'string',
            },
        },
        'required': ['prompt'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.api_key = self.cfg.get('api_key') or None

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not HAS_DASHSCOPE:
            return 'Error: dashscope package is not installed. Please install it with: pip install dashscope'

        params = self._verify_json_format_args(params)
        prompt = params['prompt']

        if self.api_key:
            dashscope.api_key = self.api_key

        from dashscope import ImageSynthesis

        response = ImageSynthesis.call(
            model='wanx-v1',
            prompt=prompt,
            api_key=self.api_key,
        )

        if response.status_code == 200:
            images = response.output.get('images', [])
            if images:
                return f"Image generated successfully: {images[0]['url']}"
            return "Image generated but no URL returned"
        else:
            return f"Error: {response.message}"


# Alias for the tool name used in tests
ImageGenerate = ImageGen
