"""基于 case 与 edit-unit 元信息的轻量分类器。

分类结果用于：
- 在 prompt 中提示模型关注结构性变化（如跨文件、类型定义、宏修改）；
- 为后续 patch 生成提供更明确的任务语境。
"""

from typing import List

from .models import CaseClassification, EditUnit, KernelCaseBundle

CONCURRENCY_KEYWORDS = ('rcu', 'race', 'stall', 'locking', 'lock', 'refcount', 'use-after-free')


def classify_case(bundle: KernelCaseBundle, edit_units: List[EditUnit]) -> CaseClassification:
    """产出 case 主类型、标签与解释原因。"""
    labels = []
    reasons = []
    message = bundle.commit_message.lower()

    if len(bundle.changed_files) > 1:
        labels.append('cross_file')
        reasons.append(f'Community fix touches {len(bundle.changed_files)} files.')
    else:
        labels.append('single_file')

    if len(edit_units) > 1:
        labels.append('multi_edit_unit')
        reasons.append(f'Community patch splits into {len(edit_units)} edit units.')
    elif edit_units:
        labels.append('single_edit_unit')

    if any(unit.operation == 'add_block' for unit in edit_units):
        labels.append('adds_block')
        reasons.append('At least one edit unit introduces a new top-level block.')

    if any(unit.block_kind in {'struct', 'union', 'enum', 'typedef'} for unit in edit_units):
        labels.append('type_definition_change')
        reasons.append('Type definitions are changed or introduced.')

    if any(unit.block_kind == 'macro' for unit in edit_units):
        labels.append('macro_change')
        reasons.append('Macro definitions are changed or introduced.')

    if any(keyword in message for keyword in CONCURRENCY_KEYWORDS):
        labels.append('concurrency_sensitive')
        reasons.append('Commit message references concurrency or RCU semantics.')

    # 根据标签优先级归并为一个 primary_kind，便于 prompt 直接消费。
    primary_kind = 'single_function_patch'
    if 'cross_file' in labels:
        primary_kind = 'cross_file_patch'
    elif 'adds_block' in labels or 'type_definition_change' in labels:
        primary_kind = 'block_structure_patch'
    elif 'concurrency_sensitive' in labels or 'multi_edit_unit' in labels:
        primary_kind = 'coordinated_logic_patch'

    return CaseClassification(primary_kind=primary_kind, labels=labels, reasons=reasons)
