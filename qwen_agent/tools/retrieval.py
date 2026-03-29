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
from qwen_agent.tools.doc_parser import DocParser


@register_tool('retrieval')
class Retrieval(BaseTool):
    """Retrieval tool that finds relevant content from documents based on a query."""

    description = 'Retrieve relevant content from documents based on a query. It parses documents and searches for relevant information.'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'description': 'The search query to find relevant content',
                'type': 'string',
            },
            'files': {
                'description': 'List of file URLs to search in',
                'type': 'array',
                'items': {'type': 'string'},
            },
            'top_k': {
                'description': 'Maximum number of results to return',
                'type': 'integer',
                'default': 3,
            },
        },
        'required': ['query'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.parser = DocParser(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params.get('query', '')
        files = params.get('files', [])
        top_k = params.get('top_k', 3)

        if not query:
            return 'Error: No query provided'

        if not files:
            return 'Error: No files provided for retrieval'

        results = []
        query_lower = query.lower()

        for file_url in files:
            try:
                # Parse the document
                content = self.parser.call({'url': file_url})

                if content.startswith('Error'):
                    continue

                # Simple keyword-based retrieval
                content_lower = content.lower()
                if query_lower in content_lower:
                    idx = content_lower.find(query_lower)
                    start = max(0, idx - 200)
                    end = min(len(content), idx + 200)
                    snippet = content[start:end]

                    # Count occurrences for ranking
                    count = content_lower.count(query_lower)
                    results.append({
                        'file': file_url,
                        'score': count,
                        'snippet': snippet,
                    })
            except Exception as e:
                continue

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]

        if not results:
            return 'No relevant content found'

        output = []
        for r in results:
            output.append(f"[File]: {r['file']}\n[Relevant content]:\n{r['snippet']}\n")

        return '\n'.join(output)
