from __future__ import annotations

import json
from typing import Iterable, List, Optional

from qwen_agent.kernel_patch.models import CodeSnippet, FixPlan, PatchHunk, RetrievalReport, RootCauseReport


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


def build_planner_prompt(decoder_report: RootCauseReport,
                         origin_hunks: List[PatchHunk],
                         target_snippets: Iterable[CodeSnippet],
                         retrieval_report: Optional[RetrievalReport] = None,
                         previous_apply_error: str = '') -> str:
    planner_payload = {
        'decoder_report': decoder_report.model_dump(),
        'origin_hunks': [h.model_dump() for h in origin_hunks],
        'target_snippets': [s.model_dump() for s in target_snippets],
        'retrieval_report': retrieval_report.model_dump() if retrieval_report is not None else None,
        'previous_apply_error': previous_apply_error,
    }
    return f'''你是 Linux 内核补丁适配 Planner。请只输出一个 JSON 对象，不要输出 Markdown。

任务：
1. 根据 decoder_report、target_snippets 和 retrieval_report，为本地仓规划最可能的修改点。
2. 每个 planned_hunk 必须精确指向一个 file_path，并解释为什么命中该代码块。
3. 若社区 patch 的 helper、宏或结构体在本地不存在，adaptation_strategy 必须说明替代方案。
4. 必须优先参考 retrieval_report 中的 hunk_coverages、entity_coverages 和 missing_entities。
5. 若某个 hunk 未覆盖，必须在 adaptation_notes 或 unresolved_risks 中明确说明。

输出 JSON 字段必须包含：
- summary
- candidate_files
- planned_hunks
- adaptation_notes
- unresolved_risks

其中 planned_hunks 每项必须包含：
- file_path
- change_type
- rationale
- anchor
- target_snippet
- adaptation_strategy

规划要求补充：
- 优先使用 retrieval_report.hunk_coverages 中已经覆盖的证据定位修改点。
- 对 retrieval_report.missing_entities 中的新增 helper/宏/结构体，必须说明在旧版本中的替代策略。
- 如果 target_snippets 中没有足够证据，不要假装已经定位成功，应明确记录风险。

规划输入：
{_json_block(planner_payload)}
'''


def build_solver_prompt(decoder_report: RootCauseReport,
                        fix_plan: FixPlan,
                        origin_patch: str,
                        target_snippets: Iterable[CodeSnippet],
                        retrieval_report: Optional[RetrievalReport] = None,
                        previous_apply_error: str = '') -> str:
    solver_payload = {
        'decoder_report': decoder_report.model_dump(),
        'fix_plan': fix_plan.model_dump(),
        'target_snippets': [s.model_dump() for s in target_snippets],
        'retrieval_report': retrieval_report.model_dump() if retrieval_report is not None else None,
        'previous_apply_error': previous_apply_error,
    }
    return f'''你是 Linux 内核补丁 Solver。

要求：
1. 只输出最终 unified diff patch 文本，不要输出解释，不要使用 Markdown 代码块。
2. patch 必须针对 target_snippets 所代表的本地仓代码。
3. 若上一次 git apply --check 失败，必须修正路径、上下文或 hunk。
4. patch 应尽量保持社区修复语义，但允许按 fix_plan.adaptation_notes 做跨版本适配。
5. 必须参考 retrieval_report，优先在已覆盖的 hunk 和 entity 证据上生成 patch。
6. 对 retrieval_report.missing_entities 中的新增 helper/宏/结构体，不要直接照抄上游定义，除非 target_snippets 中已有充分证据；优先生成旧版本等价实现。
7. 如果 retrieval_report 显示某个 hunk 未覆盖，输出的 patch 必须更加保守，优先修改已覆盖代码块。

规划输入：
{_json_block(solver_payload)}

参考 origin patch：
{origin_patch}
'''


def build_agentic_retriever_prompt(decoder_report: RootCauseReport,
                                   origin_hunks: List[PatchHunk],
                                   changed_files: List[str],
                                   origin_patch: str,
                                   repo: str,
                                   max_tool_calls: int = 6) -> str:
    payload = {
        'repo': repo,
        'decoder_report': decoder_report.model_dump(),
        'origin_hunks': [h.model_dump() for h in origin_hunks],
        'changed_files': changed_files,
        'origin_patch_excerpt': origin_patch[:12000],
    }
    return f'''你是 Linux 内核补丁检索 Retriever。你必须通过 MCP 工具自己搜索和读取目标仓代码，不允许假设。

目标：
1. 使用 `kernel_repo-search_code`、`kernel_repo-read_range` 自主检索。
2. 为每个 origin hunk 找到本地证据，输出 hunk_coverages。
3. 对社区 patch 新增的 helper、宏、结构体、include 做存在性检查，输出 entity_coverages 和 missing_entities。
4. 最终只输出一个 JSON 对象，不要输出 Markdown，不要输出解释。

输出 JSON 字段必须包含：
- snippets
- hunk_coverages
- entity_coverages
- missing_entities
- added_macros
- added_structs
- added_functions
- added_includes

规则：
- 只检索 changed_files 对应文件。
- 你必须只在 changed_files 范围内检索，不允许搜索或读取 changed_files 之外的文件。
- 前 2 次工具调用必须优先围绕 changed_files 和 impacted_functions，不允许一开始先查新增 struct/macro/helper。
- 第 1 优先动作：如果 impacted_functions 非空，先在 changed_files 内对 impacted_functions 做 `kernel_repo-search_code()` 或直接 `kernel_repo-read_range()` 读取函数附近代码。
- 第 2 优先动作：继续在 changed_files 内读取同一函数或同一文件的关键上下文，确认真实修改点。
- 只有在 changed_files 和 impacted_functions 已经查过之后，才允许在 changed_files 内继续检查新增 struct/macro/helper；仍然不允许跨文件扩展搜索。
- 若同名实体在本地不存在，明确记入 missing_entities。
- 只有在 read_range 读到具体上下文后，才能把该证据写进 snippets 或 coverages。
- 避免无效调用：不要重复查询同一 symbol/kind/path，除非上一次结果不足以定位 hunk。
- 避免过早扩散：如果 changed_files 中已经找到高质量证据，不要继续扩大搜索范围。
- 工具调用预算上限：最多 {max_tool_calls} 次。到达预算前必须尽快整理已有证据并输出 JSON。
- 不要无限搜索；当每个 hunk 都已有足够证据时停止。

输入：
{_json_block(payload)}
'''
