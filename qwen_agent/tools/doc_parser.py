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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import save_url_to_local_work_dir


@register_tool('doc_parser')
class DocParser(BaseTool):
    """Parse documents (PDF, DOCX, etc.) and extract text content."""

    description = 'Parse documents from URLs and extract their text content. Supports PDF, DOCX, and other document formats.'
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
        self.work_dir = self.cfg.get('work_dir', '/tmp/doc_parser')
        os.makedirs(self.work_dir, exist_ok=True)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        url = params['url']

        if not HAS_REQUESTS:
            return 'Error: requests package is not installed'

        try:
            # Download the file
            local_path = save_url_to_local_work_dir(url, self.work_dir)

            # Determine file type and parse
            ext = os.path.splitext(local_path)[1].lower()

            if ext == '.pdf':
                return self._parse_pdf(local_path)
            elif ext in ['.docx', '.doc']:
                return self._parse_docx(local_path)
            elif ext == '.txt':
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                return f'Unsupported file type: {ext}'

        except Exception as e:
            return f'Error parsing document: {str(e)}'

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
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text_parts.append(page.extract_text() or '')
                return '\n'.join(text_parts)
            except ImportError:
                return 'Error: Neither PyPDF2 nor pdfplumber is installed'

    def _parse_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except ImportError:
            return 'Error: python-docx is not installed'
