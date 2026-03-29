from __future__ import annotations

import json
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import json5

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.tools import MCPManager
from qwen_agent.tools.base import BaseTool

from .git_tools import summarize_text
from .models import CodeSearchHit, CodeSnippet
from .trace import TraceRecorder


class KernelRepoMCPClient:

    def __init__(self, trace: TraceRecorder, server_name: str = 'kernel_repo'):
        self.trace = trace
        self.server_name = server_name
        self._tool_map = self._init_tools()

    def _init_tools(self):
        config = build_kernel_repo_mcp_config(self.server_name)
        tools = MCPManager().initConfig(config)
        return {tool.name: tool for tool in tools}

    def _call_tool(self, tool_suffix: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = f'{self.server_name}-{tool_suffix}'
        tool = self._tool_map[tool_name]
        self.trace.record_event('tool', 'mcp_tool_start', {'tool_name': tool_name, 'arguments': arguments})
        raw_result = tool.call(json.dumps(arguments, ensure_ascii=False))
        self.trace.record_event('tool', 'mcp_tool_finish',
                                {'tool_name': tool_name, 'result': summarize_text(raw_result, limit=800)})
        payload = json.loads(raw_result)
        if not payload.get('ok', True):
            raise RuntimeError(payload.get('error', f'MCP tool failed: {tool_name}'))
        return payload

    def show_commit_patch(self, repo: str, commit_id: str) -> Dict[str, Any]:
        return self._call_tool('show_commit_patch', {'repo': repo, 'commit_id': commit_id})

    def resolve_parent_commit(self, repo: str, commit_id: str) -> str:
        return self._call_tool('resolve_parent_commit', {'repo': repo, 'commit_id': commit_id})['parent_commit']

    def search_code(self,
                    repo: str,
                    pattern: str,
                    path_glob: Optional[str] = None,
                    mode: str = 'rg') -> List[CodeSearchHit]:
        payload = self._call_tool('search_code', {
            'repo': repo,
            'pattern': pattern,
            'path_glob': path_glob,
            'mode': mode,
        })
        return [CodeSearchHit.model_validate(item) for item in payload.get('hits', [])]

    def read_range(self, repo: str, path: str, start_line: int, end_line: int, reason: str) -> CodeSnippet:
        payload = self._call_tool('read_range', {
            'repo': repo,
            'path': path,
            'start_line': start_line,
            'end_line': end_line,
        })
        return CodeSnippet(file_path=payload['path'],
                           start_line=payload['start_line'],
                           end_line=payload['end_line'],
                           content=payload['content'],
                           reason=reason)

    def read_file(self, repo: str, path: str) -> Dict[str, Any]:
        return self._call_tool('read_file', {'repo': repo, 'path': path})

    def write_file(self, repo: str, path: str, content: str) -> Dict[str, Any]:
        return self._call_tool('write_file', {'repo': repo, 'path': path, 'content': content})

    def replace_in_file(self,
                        repo: str,
                        path: str,
                        old_text: str,
                        new_text: str,
                        expected_count: int = 1) -> Dict[str, Any]:
        return self._call_tool('replace_in_file', {
            'repo': repo,
            'path': path,
            'old_text': old_text,
            'new_text': new_text,
            'expected_count': expected_count,
        })

    def replace_lines(self, repo: str, path: str, start_line: int, end_line: int, new_text: str) -> Dict[str, Any]:
        return self._call_tool('replace_lines', {
            'repo': repo,
            'path': path,
            'start_line': start_line,
            'end_line': end_line,
            'new_text': new_text,
        })

    def insert_before(self,
                      repo: str,
                      path: str,
                      anchor_text: str,
                      new_text: str,
                      expected_count: int = 1) -> Dict[str, Any]:
        return self._call_tool('insert_before', {
            'repo': repo,
            'path': path,
            'anchor_text': anchor_text,
            'new_text': new_text,
            'expected_count': expected_count,
        })

    def insert_after(self,
                     repo: str,
                     path: str,
                     anchor_text: str,
                     new_text: str,
                     expected_count: int = 1) -> Dict[str, Any]:
        return self._call_tool('insert_after', {
            'repo': repo,
            'path': path,
            'anchor_text': anchor_text,
            'new_text': new_text,
            'expected_count': expected_count,
        })

    def replace_near_anchor(self,
                            repo: str,
                            path: str,
                            anchor_text: str,
                            start_offset: int,
                            end_offset: int,
                            new_text: str,
                            expected_anchor_count: int = 1) -> Dict[str, Any]:
        return self._call_tool('replace_near_anchor', {
            'repo': repo,
            'path': path,
            'anchor_text': anchor_text,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'new_text': new_text,
            'expected_anchor_count': expected_anchor_count,
        })

    def list_files(self, repo: str, path_glob: Optional[str] = None) -> Dict[str, Any]:
        return self._call_tool('list_files', {'repo': repo, 'path_glob': path_glob})

    def run_git(self, repo: str, args: Sequence[str]) -> Dict[str, Any]:
        return self._call_tool('run_git', {'repo': repo, 'args': list(args)})

    def run_command(self, repo: str, command: str, timeout_sec: int = 30) -> Dict[str, Any]:
        return self._call_tool('run_command', {'repo': repo, 'command': command, 'timeout_sec': timeout_sec})


def build_kernel_repo_mcp_config(server_name: str = 'kernel_repo') -> Dict[str, Any]:
    server_script = str((Path(__file__).resolve().parents[2] / 'scripts' / 'kernel_repo_mcp_server.py'))
    return {
        'mcpServers': {
            server_name: {
                'command': '/root/qwen-agent/.venv/bin/python',
                'args': [server_script],
            }
        }
    }


def get_bound_kernel_repo_tools(trace: TraceRecorder,
                                repo: str,
                                allowed_tool_suffixes: Optional[Sequence[str]] = None) -> List[BaseTool]:
    client = KernelRepoMCPClient(trace=trace)
    allowed_tool_suffixes = list(allowed_tool_suffixes or [
        'search_code', 'read_range', 'read_file', 'replace_in_file', 'replace_lines', 'insert_before',
        'insert_after', 'replace_near_anchor', 'write_file', 'list_files', 'run_git', 'run_command'
    ])

    class BoundSearchCodeTool(BaseTool):
        name = 'search_code'
        description = 'Search literal code text in the repository. Do not use regex patterns.'
        parameters = {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string'
                },
                'path_glob': {
                    'type': 'string'
                },
                'mode': {
                    'type': 'string',
                    'enum': ['rg', 'git_grep']
                },
            },
            'required': ['pattern']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            pattern = args['pattern']
            if any(token in pattern for token in ('\\*', '.*', '^', '$')):
                raise RuntimeError(f'search_code only accepts literal text patterns: {pattern}')
            hits = client.search_code(repo=repo,
                                      pattern=pattern,
                                      path_glob=args.get('path_glob'),
                                      mode=args.get('mode', 'rg'))
            return json.dumps({'hits': [hit.model_dump() for hit in hits]}, ensure_ascii=False)

    class BoundReadRangeTool(BaseTool):
        name = 'read_range'
        description = 'Read a specific line range from a repository file.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string'
                },
                'start_line': {
                    'type': 'integer'
                },
                'end_line': {
                    'type': 'integer'
                },
            },
            'required': ['path', 'start_line', 'end_line']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            snippet = client.read_range(repo=repo,
                                        path=args['path'],
                                        start_line=int(args['start_line']),
                                        end_line=int(args['end_line']),
                                        reason=f'agentic:{args["path"]}')
            return json.dumps(snippet.model_dump(), ensure_ascii=False)

    class BoundReadFileTool(BaseTool):
        name = 'read_file'
        description = 'Read a full file from the repository.'
        parameters = {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.read_file(repo=repo, path=args['path']), ensure_ascii=False)

    class BoundWriteFileTool(BaseTool):
        name = 'write_file'
        description = 'Write a full file in the repository.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string'
                },
                'content': {
                    'type': 'string'
                },
            },
            'required': ['path', 'content']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.write_file(repo=repo, path=args['path'], content=args['content']),
                              ensure_ascii=False)

    class BoundReplaceInFileTool(BaseTool):
        name = 'replace_in_file'
        description = 'Replace a literal text block inside a file.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string'
                },
                'old_text': {
                    'type': 'string'
                },
                'new_text': {
                    'type': 'string'
                },
                'expected_count': {
                    'type': 'integer'
                },
            },
            'required': ['path', 'old_text', 'new_text']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.replace_in_file(repo=repo,
                                                     path=args['path'],
                                                     old_text=args['old_text'],
                                                     new_text=args['new_text'],
                                                     expected_count=int(args.get('expected_count', 1))),
                              ensure_ascii=False)

    class BoundReplaceLinesTool(BaseTool):
        name = 'replace_lines'
        description = 'Replace an inclusive line range inside a file.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {'type': 'string'},
                'start_line': {'type': 'integer'},
                'end_line': {'type': 'integer'},
                'new_text': {'type': 'string'},
            },
            'required': ['path', 'start_line', 'end_line', 'new_text']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.replace_lines(repo=repo,
                                                   path=args['path'],
                                                   start_line=int(args['start_line']),
                                                   end_line=int(args['end_line']),
                                                   new_text=args['new_text']),
                              ensure_ascii=False)

    class BoundInsertBeforeTool(BaseTool):
        name = 'insert_before'
        description = 'Insert text before a literal anchor in a file.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {'type': 'string'},
                'anchor_text': {'type': 'string'},
                'new_text': {'type': 'string'},
                'expected_count': {'type': 'integer'},
            },
            'required': ['path', 'anchor_text', 'new_text']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.insert_before(repo=repo,
                                                   path=args['path'],
                                                   anchor_text=args['anchor_text'],
                                                   new_text=args['new_text'],
                                                   expected_count=int(args.get('expected_count', 1))),
                              ensure_ascii=False)

    class BoundInsertAfterTool(BaseTool):
        name = 'insert_after'
        description = 'Insert text after a literal anchor in a file.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {'type': 'string'},
                'anchor_text': {'type': 'string'},
                'new_text': {'type': 'string'},
                'expected_count': {'type': 'integer'},
            },
            'required': ['path', 'anchor_text', 'new_text']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.insert_after(repo=repo,
                                                  path=args['path'],
                                                  anchor_text=args['anchor_text'],
                                                  new_text=args['new_text'],
                                                  expected_count=int(args.get('expected_count', 1))),
                              ensure_ascii=False)

    class BoundReplaceNearAnchorTool(BaseTool):
        name = 'replace_near_anchor'
        description = 'Replace a line range around the first matching anchor line in a file.'
        parameters = {
            'type': 'object',
            'properties': {
                'path': {'type': 'string'},
                'anchor_text': {'type': 'string'},
                'start_offset': {'type': 'integer'},
                'end_offset': {'type': 'integer'},
                'new_text': {'type': 'string'},
                'expected_anchor_count': {'type': 'integer'},
            },
            'required': ['path', 'anchor_text', 'start_offset', 'end_offset', 'new_text']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.replace_near_anchor(repo=repo,
                                                         path=args['path'],
                                                         anchor_text=args['anchor_text'],
                                                         start_offset=int(args['start_offset']),
                                                         end_offset=int(args['end_offset']),
                                                         new_text=args['new_text'],
                                                         expected_anchor_count=int(
                                                             args.get('expected_anchor_count', 1))),
                              ensure_ascii=False)

    class BoundListFilesTool(BaseTool):
        name = 'list_files'
        description = 'List files from the repository.'
        parameters = {'type': 'object', 'properties': {'path_glob': {'type': 'string'}}, 'required': []}

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.list_files(repo=repo, path_glob=args.get('path_glob')), ensure_ascii=False)

    class BoundRunGitTool(BaseTool):
        name = 'run_git'
        description = 'Run a git command in the repository.'
        parameters = {
            'type': 'object',
            'properties': {
                'args': {
                    'type': 'array',
                    'items': {'type': 'string'}
                },
            },
            'required': ['args']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.run_git(repo=repo, args=args['args']), ensure_ascii=False)

    class BoundRunCommandTool(BaseTool):
        name = 'run_command'
        description = 'Run a shell command in the repository.'
        parameters = {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string'
                },
                'timeout_sec': {
                    'type': 'integer'
                },
            },
            'required': ['command']
        }

        def call(self, params, **kwargs):
            args = self._verify_json_format_args(params)
            return json.dumps(client.run_command(repo=repo,
                                                 command=args['command'],
                                                 timeout_sec=int(args.get('timeout_sec', 30))),
                              ensure_ascii=False)

    registry = {
        'search_code': BoundSearchCodeTool,
        'read_range': BoundReadRangeTool,
        'read_file': BoundReadFileTool,
        'write_file': BoundWriteFileTool,
        'replace_in_file': BoundReplaceInFileTool,
        'replace_lines': BoundReplaceLinesTool,
        'insert_before': BoundInsertBeforeTool,
        'insert_after': BoundInsertAfterTool,
        'replace_near_anchor': BoundReplaceNearAnchorTool,
        'list_files': BoundListFilesTool,
        'run_git': BoundRunGitTool,
        'run_command': BoundRunCommandTool,
    }
    return [registry[name]() for name in allowed_tool_suffixes if name in registry]


def _validate_agentic_tool_usage(responses: Sequence[Message],
                                 allowed_tool_names: Sequence[str],
                                 server_name: str) -> None:
    allowed_tools = set(allowed_tool_names)
    allowed_tools.update({f'{server_name}-{name}' for name in allowed_tool_names})
    for message in responses:
        if message.role != ASSISTANT or not message.function_call:
            continue
        tool_name = message.function_call.name
        if not tool_name or not str(tool_name).strip():
            continue
        if tool_name not in allowed_tools:
            raise RuntimeError(f'Agentic stage called disallowed tool: {tool_name}')
        arguments = json5.loads(message.function_call.arguments)
        if tool_name.endswith('search_code'):
            pattern = arguments.get('pattern', '')
            if any(token in pattern for token in ('\\*', '.*', '^', '$')):
                raise RuntimeError(f'Agentic stage used regex-like pattern: {pattern}')
        if 'repo' in arguments:
            raise RuntimeError('Agentic stage must not supply repo')


def _count_agentic_tool_calls(responses: Sequence[Message]) -> int:
    return sum(
        1 for message in responses
        if message.role == ASSISTANT and message.function_call and message.function_call.name
        and str(message.function_call.name).strip())


def _extract_json_text(text: str) -> str:
    cleaned = (text or '').strip()
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, flags=re.S)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _run_agentic_stage(llm,
                       trace: TraceRecorder,
                       repo: str,
                       prompt: str,
                       system_message: str,
                       final_output_instruction: str,
                       server_name: str = 'kernel_repo',
                       allowed_tool_suffixes: Optional[Sequence[str]] = None,
                       max_tool_calls: int = 6,
                       timeout_sec: int = 300) -> str:
    stage_llm = get_chat_model(llm) if isinstance(llm, dict) else llm
    allowed_tool_suffixes = list(allowed_tool_suffixes or ['search_code', 'read_range'])
    bot = FnCallAgent(
        llm=stage_llm,
        system_message=system_message,
        function_list=get_bound_kernel_repo_tools(trace, repo=repo, allowed_tool_suffixes=allowed_tool_suffixes),
        name='kernel_stage_agent',
    )

    class StageTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise StageTimeout(f'agentic stage exceeded {timeout_sec}s')

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(timeout_sec)
        responses = bot.run_nonstream(messages=[Message(role='user', content=prompt)])
        signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    _validate_agentic_tool_usage(responses, allowed_tool_names=allowed_tool_suffixes, server_name=server_name)
    tool_calls = _count_agentic_tool_calls(responses)
    if tool_calls > max_tool_calls:
        raise RuntimeError(f'Agentic stage exceeded tool-call budget: {tool_calls} > {max_tool_calls}')

    convergence_messages = [Message(role='system', content=system_message)]
    convergence_messages.append(
        Message(role='user', content='The tool-using stage above has finished. Summarize it into the final required output format.'))
    convergence_messages.extend(responses)
    convergence_messages.append(Message(role='user', content=final_output_instruction))
    convergence = stage_llm.chat(messages=convergence_messages, stream=False)

    final_text = ''
    for message in convergence:
        if message.role == ASSISTANT and isinstance(message.content, str) and message.content.strip():
            final_text = message.content.strip()
    if not final_text:
        raise RuntimeError('Agentic stage did not return final output')
    final_text = _extract_json_text(final_text)
    trace.record_event('agentic_stage', 'final_output', {'summary': summarize_text(final_text, limit=1200)})
    return final_text


def run_agentic_json_stage(llm,
                           trace: TraceRecorder,
                           repo: str,
                           prompt: str,
                           system_message: str,
                           server_name: str = 'kernel_repo',
                           allowed_tool_suffixes: Optional[Sequence[str]] = None,
                           max_tool_calls: int = 6,
                           timeout_sec: int = 300) -> Dict[str, Any]:
    final_text = _run_agentic_stage(
        llm=llm,
        trace=trace,
        repo=repo,
        prompt=prompt,
        system_message=system_message,
        final_output_instruction='Stop calling tools. Now return the final JSON only. '
        'Do not call any tool. Do not output Markdown. Output only one JSON object.',
        server_name=server_name,
        allowed_tool_suffixes=allowed_tool_suffixes,
        max_tool_calls=max_tool_calls,
        timeout_sec=timeout_sec,
    )
    return json5.loads(final_text)


def run_agentic_text_stage(llm,
                           trace: TraceRecorder,
                           repo: str,
                           prompt: str,
                           system_message: str,
                           server_name: str = 'kernel_repo',
                           allowed_tool_suffixes: Optional[Sequence[str]] = None,
                           max_tool_calls: int = 6,
                           timeout_sec: int = 300) -> str:
    return _run_agentic_stage(
        llm=llm,
        trace=trace,
        repo=repo,
        prompt=prompt,
        system_message=system_message,
        final_output_instruction='Stop calling tools. Now return the final output text only. '
        'Do not call any tool. Do not output Markdown fences.',
        server_name=server_name,
        allowed_tool_suffixes=allowed_tool_suffixes,
        max_tool_calls=max_tool_calls,
        timeout_sec=timeout_sec,
    )
