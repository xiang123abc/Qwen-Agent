"""从社区补丁中抽取 EditUnit。

EditUnit 是该系统的核心中间语义，描述一次“可定位的最小编辑动作”：
- 修改哪个文件、哪个符号、哪类块；
- 属于修改已有块还是新增顶层块；
- 对应 hunk 的行号范围与锚点信息。
"""

import re
from collections import OrderedDict
from typing import List, Optional

from .models import EditUnit, KernelCaseBundle

HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
STRUCT_LINE_RE = re.compile(r'^\+(typedef\s+)?(struct|union|enum)\s+([A-Za-z_]\w*)\b')
MACRO_LINE_RE = re.compile(r'^\+#define\s+([A-Za-z_]\w*)\b')
FUNCTION_LINE_RE = re.compile(r'^\+(?:static\s+)?(?:inline\s+)?[\w\s\*]+\b([A-Za-z_]\w*)\s*\([^;]*\)\s*$')
NON_TOP_LEVEL_AFTER_PLUS_RE = re.compile(r'^\+\s')
NON_SYMBOL_FUNCTIONS = {'if', 'for', 'while', 'switch', 'sizeof'}


def _infer_added_symbol(hunk_lines: List[str]) -> tuple[str, str]:
    """在无法直接定位原块时，从新增行推断候选符号。"""
    for line in hunk_lines:
        if NON_TOP_LEVEL_AFTER_PLUS_RE.match(line):
            continue
        struct_match = STRUCT_LINE_RE.match(line)
        if struct_match:
            kind = struct_match.group(2)
            return kind, struct_match.group(3)
        macro_match = MACRO_LINE_RE.match(line)
        if macro_match:
            return 'macro', macro_match.group(1)
        fn_match = FUNCTION_LINE_RE.match(line)
        if fn_match and fn_match.group(1) not in NON_SYMBOL_FUNCTIONS:
            return 'function', fn_match.group(1)
    return 'context', '<anonymous>'


def _infer_added_blocks(hunk_lines: List[str]) -> List[tuple[str, str]]:
    """从 hunk 中提取所有新增顶层块（去重后按出现顺序返回）。"""
    blocks = []
    seen = set()
    for line in hunk_lines:
        if NON_TOP_LEVEL_AFTER_PLUS_RE.match(line):
            continue
        struct_match = STRUCT_LINE_RE.match(line)
        if struct_match:
            item = (struct_match.group(2), struct_match.group(3))
        else:
            macro_match = MACRO_LINE_RE.match(line)
            if macro_match:
                item = ('macro', macro_match.group(1))
            else:
                fn_match = FUNCTION_LINE_RE.match(line)
                if fn_match and fn_match.group(1) not in NON_SYMBOL_FUNCTIONS:
                    item = ('function', fn_match.group(1))
                else:
                    continue
        if item not in seen:
            seen.add(item)
            blocks.append(item)
    return blocks


def _summarize_hunk(hunk_lines: List[str], max_lines: int = 20) -> str:
    """截取 hunk 片段，避免上下文过长压垮 prompt。"""
    snippet = hunk_lines[:max_lines]
    if len(hunk_lines) > max_lines:
        snippet.append('...')
    return '\n'.join(snippet)


def _line_payload(line: str) -> str:
    return line[1:] if line and line[0] in (' ', '+', '-') else line


def _extract_target_snippet(hunk_lines: List[str], prefix: str) -> str:
    lines = [_line_payload(line) for line in hunk_lines if line.startswith(prefix)]
    return '\n'.join(lines).rstrip()


def _extract_anchor_snippets(hunk_lines: List[str], radius: int = 2) -> tuple[str, str]:
    changed_indices = [idx for idx, line in enumerate(hunk_lines) if line.startswith(('+', '-'))]
    if not changed_indices:
        return '', ''

    first = changed_indices[0]
    last = changed_indices[-1]
    before_lines = []
    after_lines = []

    for idx in range(first - 1, -1, -1):
        if hunk_lines[idx].startswith(' '):
            before_lines.append(_line_payload(hunk_lines[idx]))
            if len(before_lines) >= radius:
                break
    before_lines.reverse()

    for idx in range(last + 1, len(hunk_lines)):
        if hunk_lines[idx].startswith(' '):
            after_lines.append(_line_payload(hunk_lines[idx]))
            if len(after_lines) >= radius:
                break

    return '\n'.join(before_lines).rstrip(), '\n'.join(after_lines).rstrip()


def parse_edit_units(bundle: KernelCaseBundle, repo_manager, worktree_path) -> List[EditUnit]:
    """解析社区 patch，构建 EditUnit 列表。"""
    units = OrderedDict()

    def upsert_unit(path: str,
                    operation: str,
                    block_kind: str,
                    symbol: str,
                    old_start: int,
                    old_count: int,
                    new_start: int,
                    new_count: int,
                    anchor_before: str,
                    anchor_after: str,
                    target_snippet: str,
                    before_snippet: str,
                    after_snippet: str,
                    patch_excerpt: str,
                    notes: List[str]):
        """插入或合并同键 EditUnit，避免同一语义片段重复。"""
        key = (path, operation, block_kind, symbol, anchor_before, anchor_after)
        if key not in units:
            unit_id = f'{len(units) + 1:02d}:{path}:{symbol}'
            units[key] = EditUnit(unit_id=unit_id,
                                  path=path,
                                  operation=operation,
                                  block_kind=block_kind,
                                  symbol=symbol,
                                  old_start=old_start,
                                  old_count=old_count,
                                  new_start=new_start,
                                  new_count=new_count,
                                  anchor_before=anchor_before,
                                  anchor_after=anchor_after,
                                  target_snippet=target_snippet,
                                  before_snippet=before_snippet,
                                  after_snippet=after_snippet,
                                  patch_excerpt=patch_excerpt,
                                  notes=list(notes))
        else:
            existing = units[key]
            existing.old_start = min(existing.old_start, old_start)
            existing.new_start = min(existing.new_start, new_start)
            existing.old_count += old_count
            existing.new_count += new_count
            if target_snippet and not existing.target_snippet:
                existing.target_snippet = target_snippet
            if before_snippet and not existing.before_snippet:
                existing.before_snippet = before_snippet
            if after_snippet and not existing.after_snippet:
                existing.after_snippet = after_snippet
            existing.patch_excerpt += '\n\n' + patch_excerpt
            existing.notes.extend(note for note in notes if note not in existing.notes)

    current_path: Optional[str] = None
    lines = bundle.community_patch.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            parts = line.split()
            current_path = parts[2][2:] if len(parts) >= 3 and parts[2].startswith('a/') else None
            i += 1
            continue
        if current_path is None:
            i += 1
            continue
        match = HUNK_RE.match(line)
        if not match:
            i += 1
            continue

        old_start = int(match.group(1))
        old_count = int(match.group(2) or '1')
        new_start = int(match.group(3))
        new_count = int(match.group(4) or '1')
        # 收集当前 hunk 的正文行，直到下一个 hunk 或下一个文件 diff。
        j = i + 1
        hunk_lines = []
        while j < len(lines) and not lines[j].startswith('diff --git ') and not HUNK_RE.match(lines[j]):
            hunk_lines.append(lines[j])
            j += 1

        removed_lines = [hunk_line for hunk_line in hunk_lines if hunk_line.startswith('-')]
        target_snippet = _extract_target_snippet(hunk_lines, '-')
        before_snippet, after_snippet = _extract_anchor_snippets(hunk_lines)
        patch_excerpt = '@@ -{0},{1} +{2},{3} @@\n{4}'.format(
            old_start, old_count, new_start, new_count, _summarize_hunk(hunk_lines))

        # 依据“父树代码块索引”寻找与该 hunk 行域重叠的原始块。
        index = repo_manager.get_block_index(worktree_path, current_path)
        old_end = old_start + max(old_count - 1, 0)
        overlapping_blocks = [
            block for block in index.blocks
            if old_count > 0 and not (block.end_line < old_start or block.start_line > old_end)
        ]
        if old_count == 0:
            block = repo_manager.locate_block(worktree_path, current_path, old_start)
            if block is not None:
                overlapping_blocks = [block]

        # 1) 有删除且能命中原块：判定为 modify_block。
        if removed_lines and overlapping_blocks:
            for block in overlapping_blocks:
                before, after = repo_manager.nearest_blocks(worktree_path, current_path, block.start_line)
                upsert_unit(current_path,
                            'modify_block',
                            block.kind,
                            block.name,
                            old_start,
                            old_count,
                            new_start,
                            new_count,
                            before.name if before else '',
                            after.name if after else '',
                            target_snippet,
                            before_snippet,
                            after_snippet,
                            patch_excerpt,
                            [])
        # 2) 有删除但没有命中原块：回退为基于新增内容的符号推断。
        elif removed_lines:
            inferred_kind, inferred_symbol = _infer_added_symbol(hunk_lines)
            before, after = repo_manager.nearest_blocks(worktree_path, current_path, old_start)
            upsert_unit(current_path,
                        'modify_block',
                        inferred_kind,
                        inferred_symbol,
                        old_start,
                        old_count,
                        new_start,
                        new_count,
                        before.name if before else '',
                        after.name if after else '',
                        target_snippet,
                        before_snippet,
                        after_snippet,
                        patch_excerpt,
                        ['No overlapping block found; using inferred symbol.'])

        # 3) 识别新增顶层块（结构体、宏、函数等）。
        added_blocks = _infer_added_blocks(hunk_lines)
        for block_kind, symbol in added_blocks:
            if any(block.name == symbol for block in overlapping_blocks):
                continue
            before, after = repo_manager.nearest_blocks(worktree_path, current_path, old_start)
            notes = ['Pure addition hunk; treat as new top-level block.'] if not removed_lines else [
                'Hunk also introduces a new top-level block.'
            ]
            upsert_unit(current_path,
                        'add_block',
                        block_kind,
                        symbol,
                        old_start,
                        old_count,
                        new_start,
                        new_count,
                        before.name if before else '',
                        after.name if after else '',
                        target_snippet,
                        before_snippet,
                        after_snippet,
                        patch_excerpt,
                        notes)

        # 4) 纯插入场景下兜底：即使识别不到明确块，也记录 add_block 语义。
        if not removed_lines and not added_blocks and not overlapping_blocks:
            inferred_kind, inferred_symbol = _infer_added_symbol(hunk_lines)
            before, after = repo_manager.nearest_blocks(worktree_path, current_path, old_start)
            upsert_unit(current_path,
                        'add_block',
                        inferred_kind,
                        inferred_symbol,
                        old_start,
                        old_count,
                        new_start,
                        new_count,
                        before.name if before else '',
                        after.name if after else '',
                        target_snippet,
                        before_snippet,
                        after_snippet,
                        patch_excerpt,
                        ['Target symbol missing in parent tree; treat as insertion.'])
        i = j
    return list(units.values())


def summarize_edit_units(units: List[EditUnit]) -> str:
    """把 EditUnit 列表转成可直接放入 prompt 的摘要文本。"""
    if not units:
        return 'No edit units identified.'
    sections = []
    for unit in units:
        notes = '; '.join(unit.notes) if unit.notes else 'none'
        sections.append(
            f'- {unit.unit_id}: {unit.operation} {unit.block_kind} `{unit.symbol}` in `{unit.path}` '
            f'(anchors: before=`{unit.anchor_before or "-"}`, after=`{unit.anchor_after or "-"}`; notes: {notes})'
        )
    return '\n'.join(sections)


def build_prefetched_context(bundle: KernelCaseBundle, repo_manager, worktree_path, units: List[EditUnit]) -> str:
    """为 patch 生成阶段预取上下文。

    对 modify_block 读取当前父树中的目标块；
    对 add_block 读取锚点附近插入上下文；
    对类型/宏变更补充相关定义，减少模型二次检索成本。
    """
    sections = []
    for unit in units:
        sections.append(f'### Edit Unit {unit.unit_id}')
        sections.append(unit.patch_excerpt)
        if unit.operation == 'modify_block':
            block_text = repo_manager.read_block(worktree_path, unit.path, symbol=unit.symbol)
            sections.append('Current parent block:')
            sections.append(block_text)
        else:
            anchor_text = repo_manager.read_insertion_context(worktree_path, unit.path, unit.anchor_before, unit.anchor_after)
            sections.append('Current parent insertion context:')
            sections.append(anchor_text)
        if unit.block_kind in {'struct', 'union', 'enum', 'typedef'}:
            sections.append('Related type definitions:')
            sections.append(repo_manager.find_type_definition(worktree_path, unit.symbol))
        if unit.block_kind == 'macro':
            sections.append('Related macro definitions:')
            sections.append(repo_manager.find_macro_definition(worktree_path, unit.symbol))
    return '\n\n'.join(sections) if sections else repo_manager.current_hunk_context(bundle, worktree_path)
