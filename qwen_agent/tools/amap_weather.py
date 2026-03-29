# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an " "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Union

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('amap_weather')
class AmapWeather(BaseTool):
    """Query weather information using Amap (Gaode) Weather API."""

    description = '查询天气预报信息，输入城市名称（需要包含"市"字，如"北京市"），返回该城市的天气信息。'
    parameters = {
        'type': 'object',
        'properties': {
            'location': {
                'description': '城市名称，需要包含"市"字，例如"北京市"、"杭州市"',
                'type': 'string',
            },
        },
        'required': ['location'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.api_key = self.cfg.get('amap_key') or self.cfg.get('api_key') or None

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not HAS_REQUESTS:
            return 'Error: requests package is not installed.'

        params = self._verify_json_format_args(params)
        location = params['location']

        if not self.api_key:
            return 'Error: Amap API key is not configured.'

        # Clean location name (remove "市" suffix for API call)
        city = location.replace('市', '').replace('区', '').replace('县', '')

        url = 'https://restapi.amap.com/v3/weather/weatherInfo'
        params_dict = {
            'key': self.api_key,
            'city': city,
            'extensions': 'base',
        }

        try:
            import json
            response = requests.get(url, params=params_dict, timeout=10)
            data = response.json()

            if data.get('status') == '1':
                lives = data.get('lives', [])
                if lives:
                    live = lives[0]
                    result = (
                        f"城市：{live.get('city', '未知')}\n"
                        f"天气：{live.get('weather', '未知')}\n"
                        f"温度：{live.get('temperature', '未知')}℃\n"
                        f"风向：{live.get('winddirection', '未知')}\n"
                        f"风力：{live.get('windpower', '未知')}级\n"
                        f"湿度：{live.get('humidity', '未知')}%\n"
                        f"更新时间：{live.get('reporttime', '未知')}"
                    )
                    return result
            return f"查询失败：{data.get('info', '未知错误')}"
        except Exception as e:
            return f"Error: {str(e)}"
