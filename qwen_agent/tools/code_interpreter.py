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

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('code_interpreter')
class CodeInterpreter(BaseTool):
    """Execute Python code in a sandboxed environment.

    Note: This is not sandboxed and is intended for local testing only,
    not for production use.
    """

    description = 'Execute Python code and return the output. Use this to run calculations, data processing, or other programming tasks.'
    parameters = {
        'type': 'object',
        'properties': {
            'code': {
                'description': 'Python code to execute',
                'type': 'string',
            },
        },
        'required': ['code'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.timeout = self.cfg.get('timeout', 30)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # Handle raw string input (direct code string)
        if isinstance(params, str) and not params.startswith('{'):
            code = params
        else:
            params = self._verify_json_format_args(params)
            code = params.get('code', '')

        if not code:
            return 'Error: No code provided'

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        result_lines = []

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, {'__name__': '__main__'})

            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()

            if stdout_output:
                result_lines.append(stdout_output)
            if stderr_output:
                result_lines.append(f"[stderr]: {stderr_output}")

            if not result_lines:
                result_lines.append("Code executed successfully (no output)")

            return '\n'.join(result_lines)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            return f"Error executing code:\n{error_type}: {error_msg}"
