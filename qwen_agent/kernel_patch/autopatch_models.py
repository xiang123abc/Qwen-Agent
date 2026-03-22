"""Closed-loop autopatch 数据模型。"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from .models import CompileValidationResult, PatchCandidate

IDENT_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b')
STOPWORDS = {
    'linux',
    'kernel',
    'crash',
    'panic',
    'error',
    'issue',
    'invalid',
    'failure',
    'failed',
    'warning',
    'patch',
    'check',
    'read',
    'write',
    'stack',
    'trace',
    'pointer',
    'return',
    'value',
    'commit',
    'function',
    'struct',
    'null',
}


@dataclass
class VulnerabilityInput:
    """漏洞输入的统一表示。"""

    title: str = ''
    description: str = ''
    crash_log: str = ''
    reproducer: str = ''
    subsystem_hints: List[str] = field(default_factory=list)
    file_hints: List[str] = field(default_factory=list)
    symbol_hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def narrative(self) -> str:
        sections = []
        if self.title.strip():
            sections.append(f'Title: {self.title.strip()}')
        if self.description.strip():
            sections.append(f'Description:\n{self.description.strip()}')
        if self.crash_log.strip():
            sections.append(f'Crash Log:\n{self.crash_log.strip()}')
        if self.reproducer.strip():
            sections.append(f'Reproducer:\n{self.reproducer.strip()}')
        if self.subsystem_hints:
            sections.append(f'Subsystems: {", ".join(self.subsystem_hints)}')
        if self.file_hints:
            sections.append(f'File Hints: {", ".join(self.file_hints)}')
        if self.symbol_hints:
            sections.append(f'Symbol Hints: {", ".join(self.symbol_hints)}')
        if self.metadata:
            sections.append(f'Metadata: {self.metadata}')
        return '\n\n'.join(sections).strip()

    def candidate_terms(self, limit: int = 24) -> List[str]:
        terms: List[str] = []
        seen = set()
        for item in self.symbol_hints + self.subsystem_hints + self.file_hints:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
        for source in (self.title, self.description, self.crash_log, self.reproducer):
            for token in IDENT_RE.findall(source or ''):
                lowered = token.lower()
                if lowered in STOPWORDS:
                    continue
                if token not in seen:
                    seen.add(token)
                    terms.append(token)
                if len(terms) >= limit:
                    return terms
        return terms


@dataclass(frozen=True)
class SearchHit:
    """文本检索命中。"""

    path: str
    line_no: int
    preview: str
    score: float = 0.0
    term: str = ''


@dataclass(frozen=True)
class CodeWindow:
    """局部代码窗口。"""

    path: str
    start_line: int
    end_line: int
    reason: str
    content: str


@dataclass(frozen=True)
class SnippetMatch:
    """局部代码片段匹配结果。"""

    path: str
    start_line: int
    end_line: int
    score: float
    matched_text: str
    strategy: str = ''


@dataclass(frozen=True)
class InsertionAnchor:
    """插入点锚。"""

    path: str
    line_no: int
    score: float
    strategy: str
    before_line: int = 0
    after_line: int = 0


@dataclass
class RetrievalCandidate:
    """候选定位点。"""

    path: str
    line_no: int
    score: float
    reason: str
    symbol: str = ''
    preview: str = ''
    supporting_hits: List[SearchHit] = field(default_factory=list)


@dataclass
class PatchEdit:
    """Patch IR 的最小编辑单元。"""

    path: str
    operation: str
    intent: str
    reason: str
    target_snippet: str = ''
    before_snippet: str = ''
    after_snippet: str = ''
    insert_snippet: str = ''
    delete_snippet: str = ''
    confidence: float = 0.0


@dataclass
class PatchPlan:
    """结构化修复计划。"""

    root_cause: str
    invariant: str
    affected_paths: List[str] = field(default_factory=list)
    retrieval_notes: List[str] = field(default_factory=list)
    edits: List[PatchEdit] = field(default_factory=list)
    compile_targets: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PatchContext:
    """给推理器的已装配上下文。"""

    candidates: List[RetrievalCandidate] = field(default_factory=list)
    windows: List[CodeWindow] = field(default_factory=list)
    summary: str = ''


@dataclass
class PatchValidationOutcome:
    """闭环 patch 验证结果。"""

    patch_found: bool
    patch_apply_ok: bool
    changed_files: List[str] = field(default_factory=list)
    diagnostics: str = ''
    generated_patch: str = ''
    compile_validation: Optional[CompileValidationResult] = None


@dataclass
class AutoPatchResult:
    """闭环 autopatch 一次运行结果。"""

    plan: PatchPlan
    context: PatchContext
    candidate: PatchCandidate
    validation: PatchValidationOutcome
    worktree_path: str
    iterations: int = 1
