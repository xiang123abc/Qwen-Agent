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

from .base import TOOL_REGISTRY, BaseTool
from .mcp_manager import MCPManager
from .storage import Storage

# Import all tools to trigger registration
try:
    from .storage import Storage
except ImportError:
    pass

try:
    from .image_gen import ImageGen, ImageGenerate
except ImportError:
    pass

try:
    from .amap_weather import AmapWeather
except ImportError:
    pass

try:
    from .code_interpreter import CodeInterpreter
except ImportError:
    pass

try:
    from .doc_parser import DocParser
except ImportError:
    pass

try:
    from .simple_doc_parser import SimpleDocParser
except ImportError:
    pass

try:
    from .keyword_search import KeywordSearch
except ImportError:
    pass

try:
    from .hybrid_search import HybridSearch
except ImportError:
    pass

try:
    from .vector_search import VectorSearch
except ImportError:
    pass

try:
    from .retrieval import Retrieval
except ImportError:
    pass

__all__ = [
    'BaseTool',
    'TOOL_REGISTRY',
    'Storage',
    'MCPManager',
    'ImageGen',
    'ImageGenerate',
    'AmapWeather',
    'CodeInterpreter',
    'DocParser',
    'SimpleDocParser',
    'KeywordSearch',
    'HybridSearch',
    'VectorSearch',
    'Retrieval',
]
