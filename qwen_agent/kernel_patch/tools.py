"""KernelPatchAgent 的函数调用工具集合。"""

import json
from typing import Dict, List, Optional, Union

from qwen_agent.tools.base import BaseTool

from .git_ops import KernelRepoManager
from .models import KernelCaseBundle


class BaseKernelTool(BaseTool):
    """kernel 工具基类：注入 repo manager、case bundle 与 worktree。"""

    def __init__(self, manager: KernelRepoManager, bundle: KernelCaseBundle, worktree_path, cfg: Optional[Dict] = None):
        self.manager = manager
        self.bundle = bundle
        self.worktree_path = worktree_path
        super().__init__(cfg)


class KernelCaseOverviewTool(BaseKernelTool):
    """返回当前 case 元信息，供模型建立全局上下文。"""
    name = 'kernel_case_overview'
    description = 'Return the current CVE case metadata, focused files, commit ids, and the worktree root used for patch generation.'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """输出当前任务最关键的结构化元数据。"""
        self._verify_json_format_args(params)
        payload = {
            'cve_id': self.bundle.case.cve_id,
            'base_commit': self.bundle.base_commit,
            'fix_commit': self.bundle.fix_commit,
            'commit_subject': self.bundle.commit_subject,
            'changed_files': self.bundle.changed_files,
            'diff_stat': self.bundle.diff_stat,
            'worktree_root': str(self.worktree_path),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class KernelSourceSearchTool(BaseKernelTool):
    """源码检索工具，支持 focused/repo 两种范围。"""
    name = 'kernel_source_search'
    description = ('Search source code in the prepared Linux worktree. '
                   'Use focused scope first so the search stays within the files touched by the community fix.')
    parameters = {
        'type': 'object',
        'properties': {
            'pattern': {
                'type': 'string',
                'description': 'The ripgrep pattern to search for.',
            },
            'scope': {
                'type': 'string',
                'description': 'Either "focused" for community-touched files or "repo" for the whole worktree.',
            },
            'paths': {
                'type': 'array',
                'items': {
                    'type': 'string'
                },
                'description': 'Optional explicit repo-relative paths to search in.',
            },
            'max_results': {
                'type': 'integer',
                'description': 'Maximum number of matches to return.',
            },
        },
        'required': ['pattern'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行正则搜索，默认优先限制在社区变更文件。"""
        params = self._verify_json_format_args(params)
        pattern = params['pattern']
        scope = params.get('scope', 'focused')
        paths = params.get('paths')
        max_results = min(int(params.get('max_results', 60)), 200)
        if not paths and scope == 'focused':
            paths = self.bundle.changed_files
        return self.manager.search_code(self.worktree_path, pattern=pattern, paths=paths, max_results=max_results)


class KernelReadFileTool(BaseKernelTool):
    """按行读取文件切片。"""
    name = 'kernel_read_file'
    description = 'Read a repo-relative file slice with line numbers from the prepared Linux worktree.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Repo-relative path to the file.',
            },
            'start_line': {
                'type': 'integer',
                'description': '1-based start line.',
            },
            'end_line': {
                'type': 'integer',
                'description': '1-based end line.',
            },
        },
        'required': ['path', 'start_line', 'end_line'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """读取指定文件区间并限制最大返回行数。"""
        params = self._verify_json_format_args(params)
        start_line = max(1, int(params['start_line']))
        end_line = min(start_line + 299, int(params['end_line']))
        return self.manager.read_file_slice(self.worktree_path, params['path'], start_line, end_line)


class KernelReadBlockTool(BaseKernelTool):
    """读取顶层代码块（函数/类型/宏等）。"""
    name = 'kernel_read_block'
    description = 'Read a top-level code block such as a function, struct, enum, typedef, global initializer, or inline helper.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Repo-relative path to the file.',
            },
            'symbol': {
                'type': 'string',
                'description': 'Block symbol name.',
            },
            'kind': {
                'type': 'string',
                'description': 'Optional block kind such as function, struct, macro, typedef, global, inline.',
            },
            'line_no': {
                'type': 'integer',
                'description': 'Optional line number to resolve to the enclosing block.',
            },
        },
        'required': ['path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """按 symbol/kind 或 line_no 定位并返回块文本。"""
        params = self._verify_json_format_args(params)
        return self.manager.read_block(self.worktree_path,
                                       params['path'],
                                       symbol=params.get('symbol'),
                                       kind=params.get('kind'),
                                       line_no=params.get('line_no'))


class KernelSymbolContextTool(BaseKernelTool):
    """符号上下文工具：先定位再扩展源码窗口。"""
    name = 'kernel_symbol_context'
    description = ('Locate a symbol or function name in the current Linux worktree and return a wider source slice '
                   'around the match. Use this before generating the final patch.')
    parameters = {
        'type': 'object',
        'properties': {
            'symbol': {
                'type': 'string',
                'description': 'Function name, variable name, or symbol to search for.',
            },
            'scope': {
                'type': 'string',
                'description': 'Either "focused" for community-touched files or "repo" for the whole worktree.',
            },
            'paths': {
                'type': 'array',
                'items': {
                    'type': 'string'
                },
                'description': 'Optional explicit repo-relative paths to search in.',
            },
            'context_before': {
                'type': 'integer',
                'description': 'How many lines before the match to include.',
            },
            'context_after': {
                'type': 'integer',
                'description': 'How many lines after the match to include.',
            },
        },
        'required': ['symbol'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """按符号获取上下文，默认 focused 作用域。"""
        params = self._verify_json_format_args(params)
        symbol = params['symbol']
        scope = params.get('scope', 'focused')
        paths = params.get('paths')
        if not paths and scope == 'focused':
            paths = self.bundle.changed_files
        context_before = min(max(int(params.get('context_before', 20)), 0), 120)
        context_after = min(max(int(params.get('context_after', 80)), 0), 240)
        return self.manager.symbol_context(self.worktree_path,
                                           symbol=symbol,
                                           paths=paths,
                                           context_before=context_before,
                                           context_after=context_after)


class KernelTypeDefinitionTool(BaseKernelTool):
    """读取类型定义。"""
    name = 'kernel_type_definition'
    description = 'Read the definition of a struct, union, enum, or typedef related to the target patch.'
    parameters = {
        'type': 'object',
        'properties': {
            'type_name': {
                'type': 'string',
                'description': 'Type name, with or without struct/union/enum prefix.',
            },
            'paths': {
                'type': 'array',
                'items': {
                    'type': 'string'
                },
                'description': 'Optional candidate files.',
            },
        },
        'required': ['type_name'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """返回 struct/union/enum/typedef 定义文本。"""
        params = self._verify_json_format_args(params)
        return self.manager.find_type_definition(self.worktree_path, params['type_name'], paths=params.get('paths'))


class KernelMacroDefinitionTool(BaseKernelTool):
    """读取宏定义。"""
    name = 'kernel_macro_definition'
    description = 'Read the definition of a macro related to the target patch.'
    parameters = {
        'type': 'object',
        'properties': {
            'macro_name': {
                'type': 'string',
                'description': 'Macro name.',
            },
            'paths': {
                'type': 'array',
                'items': {
                    'type': 'string'
                },
                'description': 'Optional candidate files.',
            },
        },
        'required': ['macro_name'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """返回目标宏的定义内容。"""
        params = self._verify_json_format_args(params)
        return self.manager.find_macro_definition(self.worktree_path, params['macro_name'], paths=params.get('paths'))


class KernelIncludeContextTool(BaseKernelTool):
    """读取源文件头部 include 区域。"""
    name = 'kernel_include_context'
    description = 'Read the include section at the top of a target source file.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Repo-relative path to the file.',
            }
        },
        'required': ['path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """返回 include 片段，便于判断头文件依赖。"""
        params = self._verify_json_format_args(params)
        return self.manager.read_include_context(self.worktree_path, params['path'])


class KernelReferencePatchTool(BaseKernelTool):
    """读取社区参考补丁（`git show -W`）。"""
    name = 'kernel_reference_patch'
    description = ('Return the community reference fix obtained via git show -W for the fixed commit. '
                   'Optionally narrow the output to a specific file.')
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Optional repo-relative file path to filter the reference patch.',
            },
            'max_chars': {
                'type': 'integer',
                'description': 'Optional output cap in characters.',
            },
        },
        'required': [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """返回参考补丁文本，可按文件过滤并截断长度。"""
        params = self._verify_json_format_args(params)
        path = params.get('path')
        max_chars = int(params.get('max_chars', 40000))
        show_text = self.manager.reference_show(self.bundle.fix_commit, paths=[path] if path else None)
        if len(show_text) > max_chars:
            show_text = show_text[:max_chars] + '\n...\n[TRUNCATED]'
        return show_text


def build_kernel_tools(manager: KernelRepoManager, bundle: KernelCaseBundle, worktree_path) -> List[BaseTool]:
    """构建并返回 KernelPatchAgent 使用的全部工具。"""
    return [
        KernelCaseOverviewTool(manager, bundle, worktree_path),
        KernelSourceSearchTool(manager, bundle, worktree_path),
        KernelReadFileTool(manager, bundle, worktree_path),
        KernelReadBlockTool(manager, bundle, worktree_path),
        KernelSymbolContextTool(manager, bundle, worktree_path),
        KernelTypeDefinitionTool(manager, bundle, worktree_path),
        KernelMacroDefinitionTool(manager, bundle, worktree_path),
        KernelIncludeContextTool(manager, bundle, worktree_path),
        KernelReferencePatchTool(manager, bundle, worktree_path),
    ]
