"""Patch case 数据加载工具。

输入文件每行格式：`CVE-xxxx-xxxxx,<fix_commit_sha>`。
"""

import re
from pathlib import Path
from typing import Iterable, List, Optional

from .models import PatchCase

CASE_RE = re.compile(r'^(CVE-\d{4}-\d+)\s*,\s*([0-9a-fA-F]{7,40})\s*$')


def iter_patch_cases(lines: Iterable[str], cve_filter: Optional[str] = None) -> List[PatchCase]:
    """从文本行中解析并去重 patch case。"""
    cve_filter = cve_filter.strip() if cve_filter else None
    cases: List[PatchCase] = []
    seen = set()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        # 非法行直接跳过，保持容错。
        match = CASE_RE.match(line)
        if not match:
            continue
        cve_id = match.group(1)
        fix_commit = match.group(2).lower()
        if cve_filter and cve_id != cve_filter:
            continue
        key = (cve_id, fix_commit)
        # 同一 `(cve_id, fix_commit)` 仅保留一次。
        if key in seen:
            continue
        seen.add(key)
        cases.append(PatchCase(cve_id=cve_id, fix_commit=fix_commit, source_line=line_no))
    return cases


def load_patch_cases(path: str, cve_filter: Optional[str] = None, limit: Optional[int] = None) -> List[PatchCase]:
    """从文件加载 case 列表，并支持 CVE 过滤与数量截断。"""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    cases = iter_patch_cases(lines, cve_filter=cve_filter)
    if limit is not None:
        return cases[:limit]
    return cases
