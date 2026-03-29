from __future__ import annotations

import json
from typing import List

from qwen_agent.kernel_patch.models import FixPlan, PatchHunk, RootCauseReport


def _json_block(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_decoder_prompt(commit_message: str,
                         changed_files: List[str],
                         origin_patch: str,
                         origin_hunks: List[PatchHunk]) -> str:
    return f'''你是 Linux 内核漏洞修复分析器。请只输出一个 JSON 对象，不要输出 Markdown。

任务：
1. 分析社区修复 commit message 和 patch。
2. 给出漏洞根因、受影响符号、修复逻辑。
3. semantic_anchors 要提炼成后续跨版本检索可用的短语或代码语义点。

输出 JSON 字段必须包含：
- vulnerability_type
- root_cause
- impacted_files
- impacted_functions
- impacted_macros
- impacted_structs
- impacted_globals
- fix_logic
- semantic_anchors
- confidence

Changed files:
{_json_block(changed_files)}

Parsed hunks:
{_json_block([h.model_dump() for h in origin_hunks])}

Commit message:
{commit_message}

Origin patch:
{origin_patch}
'''


def build_planner_agentic_prompt(decoder_report: RootCauseReport,
                                 origin_hunks: List[PatchHunk],
                                 changed_files: List[str],
                                 previous_apply_error: str = '',
                                 max_tool_calls: int = 6) -> str:
    payload = {
        'decoder_report': decoder_report.model_dump(),
        'origin_hunks': [h.model_dump() for h in origin_hunks],
        'changed_files': changed_files,
        'previous_apply_error': previous_apply_error,
    }
    return f'''你是 Linux 内核补丁适配 Planner。你必须自己调用 MCP 工具理解目标仓代码，再输出一个 JSON 对象。

工具使用规则：
- 可以使用 `search_code`、`read_range`、`read_file`、`list_files`、`run_git`、`run_command`
- 优先围绕 changed_files 和 impacted_functions 收集证据
- 最多 {max_tool_calls} 次工具调用，拿到足够证据后立刻停止

任务：
1. 在目标仓定位与 origin_hunks 语义最接近的代码块，优先检查 changed_files。
2. 给出跨版本适配计划；如果本地没有完全对应实现，要说明替代方案。
3. 为 Solver 准备最少但足够的本地代码片段。
4. 最终只输出 JSON，不要输出 Markdown。

输出 JSON 字段必须包含：
- summary
- candidate_files
- planned_hunks
- solver_snippets
- adaptation_notes
- unresolved_risks

其中 planned_hunks 每项必须包含：
- file_path
- change_type
- rationale
- anchor
- target_snippet
- adaptation_strategy

其中 solver_snippets 每项必须包含：
- file_path
- start_line
- end_line
- content
- reason

硬约束：
- `candidate_files` 必须是字符串数组，不要输出对象
- 最终回复必须是一个单独的 JSON 对象
- 不允许输出 Markdown 代码块
- 不允许输出自然语言前后缀、标题、注释

输入：
{_json_block(payload)}
'''


def build_solver_agentic_prompt(decoder_report: RootCauseReport,
                                fix_plan: FixPlan,
                                origin_patch: str,
                                changed_files: List[str],
                                previous_apply_error: str = '',
                                max_tool_calls: int = 6) -> str:
    payload = {
        'decoder_report': decoder_report.model_dump(),
        'fix_plan': fix_plan.model_dump(),
        'changed_files': changed_files,
        'previous_apply_error': previous_apply_error,
    }
    return f'''你是 Linux 内核补丁 Solver。你必须自己调用 MCP 工具在目标仓内完成修改，然后输出最终结果。

工具使用规则：
- 可以使用 `search_code`、`read_range`、`read_file`、`replace_in_file`、`replace_lines`、`replace_near_anchor`、`insert_before`、`insert_after`、`list_files`、`run_git`、`run_command`
- 默认不要使用 `write_file`；只有当文件很小且必须整体重写时才允许
- 允许直接修改仓库文件，并通过 git/diff/命令自检
- 最多 {max_tool_calls} 次工具调用，完成必要修改后立刻停止

要求：
1. 优先输出一个 JSON 对象，不要输出解释，不要输出 Markdown。
2. JSON 至少包含：
   - summary
   - status
   - modified_files
3. 优先顺序：
   - `replace_in_file`
   - `replace_lines`
   - `replace_near_anchor`
   - `insert_before` / `insert_after`
   - 最后才是 `write_file`
4. 如果你已经通过上述工具修改了文件，`status` 写 `files_modified`，`modified_files` 写文件路径数组。
5. 每次修改后优先用 `run_git` 或 `run_command` 查看最小 diff，并确认没有多余改动。
6. 也允许额外返回 `edits`，格式与旧版兼容；如果提供 `edits`，系统会按 edits 应用修改。
7. 若无法安全完成修复，返回合法 JSON，并把 `status` 写为 `failed`。
8. patch 语义应保持社区修复逻辑，但允许按 fix_plan 做跨版本适配。

输入：
{_json_block(payload)}

参考 origin patch：
{origin_patch}
'''
