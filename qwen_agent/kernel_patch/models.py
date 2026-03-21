import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + '\n...\n[TRUNCATED]\n...\n' + text[-tail:]


def extract_patch_from_response(text: str) -> Optional[str]:
    fence_matches = re.findall(r'```(?:diff|patch)?\s*\n(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
    for block in fence_matches:
        block = block.strip('\n')
        if 'diff --git ' in block or ('--- ' in block and '+++ ' in block):
            return block.rstrip() + '\n'

    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith('diff --git ') or line.startswith('--- '):
            start = index
            break
    if start is not None:
        return '\n'.join(lines[start:]).rstrip() + '\n'
    return None


def strip_patch_from_response(text: str, patch_text: Optional[str]) -> str:
    if not patch_text:
        return text.strip()
    escaped = re.escape(patch_text.rstrip())
    text = re.sub(rf'```(?:diff|patch)?\s*\n{escaped}\s*```', '', text, flags=re.DOTALL | re.IGNORECASE)
    if patch_text in text:
        text = text.replace(patch_text, '')
    return text.strip()


def _stringify_paths(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, list):
        return [_stringify_paths(item) for item in data]
    if isinstance(data, dict):
        return {key: _stringify_paths(value) for key, value in data.items()}
    return data


@dataclass(frozen=True)
class PatchCase:
    cve_id: str
    fix_commit: str
    source_line: int

    @property
    def slug(self) -> str:
        return f'{self.cve_id.lower()}__{self.fix_commit[:12]}'.replace('/', '_')


@dataclass
class KernelCaseBundle:
    case: PatchCase
    repo_root: Path
    workspace_root: Path
    base_commit: str
    fix_commit: str
    commit_subject: str
    commit_message: str
    changed_files: List[str]
    diff_stat: str
    reference_show_excerpt: str
    community_patch: str
    community_patch_excerpt: str

    @property
    def artifact_dir(self) -> Path:
        return self.workspace_root / 'artifacts' / self.case.slug


@dataclass
class PatchCandidate:
    analysis_text: str
    patch_text: Optional[str]
    raw_response: str

    @property
    def patch_found(self) -> bool:
        return bool(self.patch_text and self.patch_text.strip())


@dataclass
class PatchEvaluation:
    patch_found: bool
    patch_apply_ok: bool
    tree_match: bool
    similarity: float
    touched_files: List[str] = field(default_factory=list)
    error_message: str = ''
    remaining_diff: str = ''
    feedback_message: str = ''
    categories: List[str] = field(default_factory=list)
    generated_patch_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return _stringify_paths(asdict(self))


@dataclass
class CaseRunResult:
    case: PatchCase
    base_commit: str
    fix_commit: str
    attempts: int
    success: bool
    best_similarity: float
    artifact_dir: Path
    evaluations: List[PatchEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'case': _stringify_paths(asdict(self.case)),
            'base_commit': self.base_commit,
            'fix_commit': self.fix_commit,
            'attempts': self.attempts,
            'success': self.success,
            'best_similarity': self.best_similarity,
            'artifact_dir': str(self.artifact_dir),
            'evaluations': [evaluation.to_dict() for evaluation in self.evaluations],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
