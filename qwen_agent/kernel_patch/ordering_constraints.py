"""检测补丁与目标树之间的“语句/声明顺序”偏差。"""

import re
from typing import List

DECLARATION_RE = re.compile(
    r'^(?:const\s+)?(?:unsigned\s+|struct\s+|union\s+|enum\s+|static\s+|int\s+|char\s+|bool\s+|u\d+\s+|s\d+\s+|long\s+|short\s+|void\s+).*;'
)


def detect_ordering_constraints(diff_text: str) -> List[str]:
    """从 diff 中提取顺序相关风险标签。"""
    restore_lines = []
    avoid_lines = []
    for line in diff_text.splitlines():
        if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
            content = line[1:].strip()
            if not content:
                continue
            if line.startswith('-'):
                restore_lines.append(content)
            else:
                avoid_lines.append(content)

    categories = []
    # 同时存在“应恢复行”和“多余新增行”通常意味着语句顺序偏离。
    if restore_lines and avoid_lines:
        categories.append('statement_order_mismatch')
    if any(DECLARATION_RE.match(line) for line in restore_lines + avoid_lines):
        categories.append('declaration_order_mismatch')
    return categories
