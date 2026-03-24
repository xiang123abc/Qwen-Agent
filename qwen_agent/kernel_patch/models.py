from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PatchHunk(BaseModel):
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    anchor_lines: List[str] = Field(default_factory=list)


class CodeSearchHit(BaseModel):
    file_path: str
    line_number: int
    line_text: str
    reason: str


class CodeSnippet(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    reason: str


class CodeEdit(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    new_content: str
    rationale: str = ''


class CodeEditPlan(BaseModel):
    edits: List[CodeEdit] = Field(default_factory=list)
    summary: str = ''


class RetrievalEvidence(BaseModel):
    strategy: str
    reason: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ''
    hit_line: int = 0


class HunkCoverageReport(BaseModel):
    hunk_index: int
    file_path: str
    covered: bool = False
    evidence: List[RetrievalEvidence] = Field(default_factory=list)
    missing_anchors: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class EntityCoverageReport(BaseModel):
    entity_name: str
    entity_type: str
    found: bool = False
    evidence: List[RetrievalEvidence] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RetrievalReport(BaseModel):
    snippets: List[CodeSnippet] = Field(default_factory=list)
    hunk_coverages: List[HunkCoverageReport] = Field(default_factory=list)
    entity_coverages: List[EntityCoverageReport] = Field(default_factory=list)
    missing_entities: List[str] = Field(default_factory=list)
    added_macros: List[str] = Field(default_factory=list)
    added_structs: List[str] = Field(default_factory=list)
    added_functions: List[str] = Field(default_factory=list)
    added_includes: List[str] = Field(default_factory=list)


class RootCauseReport(BaseModel):
    vulnerability_type: str
    root_cause: str
    impacted_files: List[str] = Field(default_factory=list)
    impacted_functions: List[str] = Field(default_factory=list)
    impacted_macros: List[str] = Field(default_factory=list)
    impacted_structs: List[str] = Field(default_factory=list)
    impacted_globals: List[str] = Field(default_factory=list)
    fix_logic: List[str] = Field(default_factory=list)
    semantic_anchors: List[str] = Field(default_factory=list)
    confidence: str = 'medium'

    @field_validator('vulnerability_type', 'root_cause', 'confidence', mode='before')
    @classmethod
    def coerce_strings(cls, value):
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        return str(value)

    @field_validator(
        'impacted_files',
        'impacted_functions',
        'impacted_macros',
        'impacted_structs',
        'impacted_globals',
        'fix_logic',
        'semantic_anchors',
        mode='before')
    @classmethod
    def coerce_list_fields(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if '\n' in stripped:
                return [line.strip('-* ').strip() for line in stripped.splitlines() if line.strip()]
            return [stripped]
        return [value]


class PlannedChange(BaseModel):
    file_path: str
    change_type: str
    rationale: str
    anchor: str
    target_snippet: str
    adaptation_strategy: str

    @field_validator('file_path', 'change_type', 'rationale', 'anchor', 'target_snippet', 'adaptation_strategy',
                     mode='before')
    @classmethod
    def coerce_strings(cls, value):
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        return str(value)


class FixPlan(BaseModel):
    summary: str
    candidate_files: List[str] = Field(default_factory=list)
    planned_hunks: List[PlannedChange] = Field(default_factory=list)
    solver_snippets: List[CodeSnippet] = Field(default_factory=list)
    adaptation_notes: List[str] = Field(default_factory=list)
    unresolved_risks: List[str] = Field(default_factory=list)

    @field_validator('summary', mode='before')
    @classmethod
    def coerce_summary(cls, value):
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        return str(value)

    @field_validator('candidate_files', 'adaptation_notes', 'unresolved_risks', mode='before')
    @classmethod
    def coerce_string_lists(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if '\n' in stripped:
                return [line.strip('-* ').strip() for line in stripped.splitlines() if line.strip()]
            return [stripped]
        return [value]

    @field_validator('planned_hunks', mode='before')
    @classmethod
    def coerce_planned_hunks(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return value

    @field_validator('solver_snippets', mode='before')
    @classmethod
    def coerce_solver_snippets(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return value


class SolverAttempt(BaseModel):
    iteration: int
    apply_check_passed: bool
    patch_path: str
    apply_check_stderr: str = ''
    solver_raw_output: str = ''


class ToolTraceEntry(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    result_summary: str = ''


class StepTrace(BaseModel):
    step: str
    prompt: Optional[str] = None
    output: Any = None
    tool_calls: List[ToolTraceEntry] = Field(default_factory=list)


class KernelPatchRunResult(BaseModel):
    cve_id: str
    success: bool
    target_commit: str
    artifact_dir: str
    target_repo_path: str
    analysis_path: str
    patch_path: str
    trace_path: str
    decoder_report: RootCauseReport
    final_plan: FixPlan
    attempts: List[SolverAttempt] = Field(default_factory=list)
    summary: str
