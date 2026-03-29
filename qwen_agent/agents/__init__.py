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

from qwen_agent.agent import Agent, BasicAgent

ClosedLoopKernelPatchAgent = None
FnCallAgent = None
KernelPatchAgent = None
Assistant = None
ArticleAgent = None

try:
    from .closed_loop_kernel_patch_agent import ClosedLoopKernelPatchAgent
except ImportError:
    pass

try:
    from .fncall_agent import FnCallAgent
except ImportError:
    pass

try:
    from .kernel_patch_agent import KernelPatchAgent
except ImportError:
    pass

try:
    from .assistant import Assistant
except ImportError:
    pass

try:
    from .article_agent import ArticleAgent
except ImportError:
    pass

__all__ = [
    'Agent',
    'BasicAgent',
    'FnCallAgent',
    'KernelPatchAgent',
    'ClosedLoopKernelPatchAgent',
    'Assistant',
    'ArticleAgent',
]
