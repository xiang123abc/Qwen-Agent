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


@register_tool('hybrid_search')
class HybridSearch(BaseTool):
    """Hybrid search combining keyword and semantic vector search."""

    description = 'Search using a combination of keyword matching and semantic similarity for better results.'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'description': 'The search query',
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

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params.get('query', '')
        files = params.get('files', [])
        top_k = params.get('top_k', 5)

        if not query:
            return 'Error: No query provided'

        if not files:
            return 'Hybrid search requires a list of files to search in'

        # Simple hybrid approach: keyword matching + basic similarity
        results = []
        query_lower = query.lower()

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_lower = content.lower()

                    # Keyword score
                    keyword_count = content_lower.count(query_lower)

                    # Simple length-based similarity score
                    query_words = set(query_lower.split())
                    content_words = set(content_lower.split())
                    overlap = len(query_words & content_words)
                    similarity = overlap / max(len(query_words), 1)

                    # Combined score
                    score = keyword_count * 0.5 + similarity * 10

                    if score > 0:
                        idx = content_lower.find(query_lower)
                        start = max(0, idx - 100)
                        end = min(len(content), idx + len(query) + 100)
                        snippet = content[start:end]

                        results.append({
                            'file': file_path,
                            'score': score,
                            'snippet': snippet,
                        })
            except Exception as e:
                continue

        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]

        if not results:
            return 'No results found'

        output = []
        for i, r in enumerate(results, 1):
            output.append(f"[{i}] {r['file']} (score: {r['score']:.2f})\n{r['snippet']}\n")

        return '\n'.join(output)
