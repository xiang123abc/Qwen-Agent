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

import os
from typing import Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import save_url_to_local_work_dir


@register_tool('simple_doc_parser')
class SimpleDocParser(BaseTool):
    """A simpler document parser that extracts text from various file formats."""

    description = 'Simple document parser that extracts text from TXT, PDF, and HTML files.'
    parameters = {
        'type': 'object',
        'properties': {
            'url': {
                'description': 'URL of the document to parse',
                'type': 'string',
            },
        },
        'required': ['url'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.work_dir = self.cfg.get('work_dir', '/tmp/simple_doc_parser')
        os.makedirs(self.work_dir, exist_ok=True)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        url = params['url']

        try:
            local_path = save_url_to_local_work_dir(url, self.work_dir)
            ext = os.path.splitext(local_path)[1].lower()

            if ext == '.txt':
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif ext == '.html' or ext == '.htm':
                return self._parse_html(local_path)
            elif ext == '.pdf':
                return self._parse_pdf(local_path)
            else:
                return f'Unsupported file type: {ext}'

        except Exception as e:
            return f'Error parsing document: {str(e)}'

    def _parse_html(self, path: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n', strip=True)
        except ImportError:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Simple HTML tag removal
                import re
                clean = re.sub(r'<[^>]+>', '', content)
                return clean.strip()

    def _parse_pdf(self, path: str) -> str:
        try:
            import PyPDF2
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text())
                return '\n'.join(text_parts)
        except ImportError:
            return 'Error: PyPDF2 is not installed'
