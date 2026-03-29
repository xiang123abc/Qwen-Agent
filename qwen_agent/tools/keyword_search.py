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

from typing import Dict, List, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('keyword_search')
class KeywordSearch(BaseTool):
    """Keyword-based search tool for finding relevant documents or text."""

    description = 'Search for documents or text using keyword matching. Returns relevant results based on keyword queries.'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'description': 'The search query/keywords',
                'type': 'string',
            },
            'files': {
                'description': 'List of file URLs or paths to search in',
                'type': 'array',
                'items': {'type': 'string'},
            },
            'top_k': {
                'description': 'Maximum number of results to return',
                'type': 'integer',
                'default': 5,
            },
        },
        'required': ['query'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.index = {}  # Simple in-memory index

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params.get('query', '')
        files = params.get('files', [])
        top_k = params.get('top_k', 5)

        if not query:
            return 'Error: No query provided'

        if not files:
            return 'Keyword search requires a list of files to search in'

        results = []
        query_lower = query.lower()

        for file_path in files:
            try:
                # Simple keyword matching
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_lower = content.lower()

                    # Count keyword occurrences
                    count = content_lower.count(query_lower)

                    if count > 0:
                        # Find surrounding context
                        idx = content_lower.find(query_lower)
                        start = max(0, idx - 100)
                        end = min(len(content), idx + len(query) + 100)
                        snippet = content[start:end]

                        results.append({
                            'file': file_path,
                            'count': count,
                            'snippet': snippet,
                        })
            except Exception as e:
                continue

        # Sort by count and return top_k
        results.sort(key=lambda x: x['count'], reverse=True)
        results = results[:top_k]

        if not results:
            return 'No results found'

        output = []
        for i, r in enumerate(results, 1):
            output.append(f"[{i}] {r['file']} (matches: {r['count']})\n{r['snippet']}\n")

        return '\n'.join(output)
