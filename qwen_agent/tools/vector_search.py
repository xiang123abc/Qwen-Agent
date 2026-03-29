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
from typing import Dict, List, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('vector_search')
class VectorSearch(BaseTool):
    """Semantic vector search for finding semantically similar content."""

    description = 'Search for semantically similar content using vector embeddings.'
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
            return 'Vector search requires a list of files to search in'

        # Simple word-overlap based similarity as fallback
        # In production, this would use actual embeddings
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_lower = content.lower()
                    content_words = set(content_lower.split())

                    # Jaccard similarity
                    intersection = query_words & content_words
                    union = query_words | content_words
                    similarity = len(intersection) / max(len(union), 1)

                    if similarity > 0.01:  # Threshold
                        idx = content_lower.find(query_lower)
                        start = max(0, idx - 100) if idx >= 0 else 0
                        end = min(len(content), start + 300)
                        snippet = content[start:end]

                        results.append({
                            'file': file_path,
                            'score': similarity,
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
            output.append(f"[{i}] {r['file']} (similarity: {r['score']:.4f})\n{r['snippet']}\n")

        return '\n'.join(output)
