import re
from pathlib import Path
from typing import Iterable, List, Optional

from .models import PatchCase

CASE_RE = re.compile(r'^(CVE-\d{4}-\d+)\s*,\s*([0-9a-fA-F]{7,40})\s*$')


def iter_patch_cases(lines: Iterable[str], cve_filter: Optional[str] = None) -> List[PatchCase]:
    cve_filter = cve_filter.strip() if cve_filter else None
    cases: List[PatchCase] = []
    seen = set()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = CASE_RE.match(line)
        if not match:
            continue
        cve_id = match.group(1)
        fix_commit = match.group(2).lower()
        if cve_filter and cve_id != cve_filter:
            continue
        key = (cve_id, fix_commit)
        if key in seen:
            continue
        seen.add(key)
        cases.append(PatchCase(cve_id=cve_id, fix_commit=fix_commit, source_line=line_no))
    return cases


def load_patch_cases(path: str, cve_filter: Optional[str] = None, limit: Optional[int] = None) -> List[PatchCase]:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    cases = iter_patch_cases(lines, cve_filter=cve_filter)
    if limit is not None:
        return cases[:limit]
    return cases
