"""kernel patch 流程的数据模型定义。"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def truncate_middle(text: str, max_chars: int) -> str:
    """按“保留首尾、截断中间”的方式压缩长文本。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + '\n...\n[TRUNCATED]\n...\n' + text[-tail:]


def extract_patch_from_response(text: str) -> Optional[str]:
    """从模型输出中提取 unified diff 文本。"""
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
    """从原始响应中剥离已提取出的 patch 内容。"""
    if not patch_text:
        return text.strip()
    escaped = re.escape(patch_text.rstrip())
    text = re.sub(rf'```(?:diff|patch)?\s*\n{escaped}\s*```', '', text, flags=re.DOTALL | re.IGNORECASE)
    if patch_text in text:
        text = text.replace(patch_text, '')
    return text.strip()


def _stringify_paths(data: Any) -> Any:
    """递归把 Path 转为字符串，便于 JSON 序列化。"""
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, list):
        return [_stringify_paths(item) for item in data]
    if isinstance(data, dict):
        return {key: _stringify_paths(value) for key, value in data.items()}
    return data


@dataclass(frozen=True)
class PatchCase:
    """待修复样例：CVE + 修复提交。"""
    cve_id: str
    fix_commit: str
    source_line: int

    @property
    def slug(self) -> str:
        """稳定的 case 标识，可用于目录命名。"""
        return f'{self.cve_id.lower()}__{self.fix_commit[:12]}'.replace('/', '_')


@dataclass
class KernelCaseBundle:
    """单个 case 的运行上下文快照。"""
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
        """该 case 的产物目录路径。"""
        return self.workspace_root / 'artifacts' / self.case.slug


@dataclass(frozen=True)
class CodeBlock:
    """代码块索引条目。"""
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    signature: str = ''


@dataclass
class FileBlockIndex:
    """单文件代码块索引。"""
    path: str
    blocks: List[CodeBlock] = field(default_factory=list)


@dataclass
class EditUnit:
    """补丁编辑单元（最小可描述修改动作）。"""
    unit_id: str
    path: str
    operation: str
    block_kind: str
    symbol: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    anchor_before: str = ''
    anchor_after: str = ''
    target_snippet: str = ''
    before_snippet: str = ''
    after_snippet: str = ''
    patch_excerpt: str = ''
    resolved_start_line: int = 0
    resolved_end_line: int = 0
    alignment_score: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class CaseClassification:
    """case 语义分类结果。"""
    primary_kind: str
    labels: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """供 prompt 使用的分类摘要文本。"""
        labels = ', '.join(self.labels) if self.labels else self.primary_kind
        reasons = '; '.join(self.reasons) if self.reasons else 'No additional reasons.'
        return f'Primary kind: {self.primary_kind}\nLabels: {labels}\nReasons: {reasons}'


@dataclass
class CompileValidationResult:
    """编译验证结果。"""
    status: str
    command: str = ''
    output: str = ''
    validated_targets: List[str] = field(default_factory=list)


@dataclass
class PatchCandidate:
    """模型单次生成的候选补丁。"""
    analysis_text: str
    patch_text: Optional[str]
    raw_response: str

    @property
    def patch_found(self) -> bool:
        """是否解析到非空 patch 文本。"""
        return bool(self.patch_text and self.patch_text.strip())


@dataclass
class PatchEvaluation:
    """对候选补丁的评估结果。"""
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
    compile_validation: Optional[CompileValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化字典。"""
        return _stringify_paths(asdict(self))


@dataclass
class CaseRunResult:
    """单个 case 全部尝试的汇总结果。"""
    case: PatchCase
    base_commit: str
    fix_commit: str
    attempts: int
    success: bool
    best_similarity: float
    artifact_dir: Path
    evaluations: List[PatchEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化字典。"""
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
        """导出 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
