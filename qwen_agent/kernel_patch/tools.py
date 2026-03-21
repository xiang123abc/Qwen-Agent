import json
from typing import Dict, List, Optional, Union

from qwen_agent.tools.base import BaseTool

from .git_ops import KernelRepoManager
from .models import KernelCaseBundle


class BaseKernelTool(BaseTool):

    def __init__(self, manager: KernelRepoManager, bundle: KernelCaseBundle, worktree_path, cfg: Optional[Dict] = None):
        self.manager = manager
        self.bundle = bundle
        self.worktree_path = worktree_path
        super().__init__(cfg)


class KernelCaseOverviewTool(BaseKernelTool):
    name = 'kernel_case_overview'
    description = 'Return the current CVE case metadata, focused files, commit ids, and the worktree root used for patch generation.'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def call(self, params: Union[str, dict], **kwargs) -> str:
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
        params = self._verify_json_format_args(params)
        pattern = params['pattern']
        scope = params.get('scope', 'focused')
        paths = params.get('paths')
        max_results = min(int(params.get('max_results', 60)), 200)
        if not paths and scope == 'focused':
            paths = self.bundle.changed_files
        return self.manager.search_code(self.worktree_path, pattern=pattern, paths=paths, max_results=max_results)


class KernelReadFileTool(BaseKernelTool):
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
        params = self._verify_json_format_args(params)
        start_line = max(1, int(params['start_line']))
        end_line = min(start_line + 299, int(params['end_line']))
        return self.manager.read_file_slice(self.worktree_path, params['path'], start_line, end_line)


class KernelSymbolContextTool(BaseKernelTool):
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


class KernelReferencePatchTool(BaseKernelTool):
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
        params = self._verify_json_format_args(params)
        path = params.get('path')
        max_chars = int(params.get('max_chars', 40000))
        show_text = self.manager.reference_show(self.bundle.fix_commit, paths=[path] if path else None)
        if len(show_text) > max_chars:
            show_text = show_text[:max_chars] + '\n...\n[TRUNCATED]'
        return show_text


def build_kernel_tools(manager: KernelRepoManager, bundle: KernelCaseBundle, worktree_path) -> List[BaseTool]:
    return [
        KernelCaseOverviewTool(manager, bundle, worktree_path),
        KernelSourceSearchTool(manager, bundle, worktree_path),
        KernelReadFileTool(manager, bundle, worktree_path),
        KernelSymbolContextTool(manager, bundle, worktree_path),
        KernelReferencePatchTool(manager, bundle, worktree_path),
    ]
