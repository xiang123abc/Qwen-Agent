from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class KernelPatchLLMConfig(BaseModel):
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model_type: str = 'oai'
    generate_cfg: Dict[str, Any] = Field(default_factory=dict)

    def to_qwen_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            'model': self.model,
            'model_type': self.model_type,
            'generate_cfg': dict(self.generate_cfg),
        }
        if self.api_base:
            cfg['api_base'] = self.api_base
            cfg['model_server'] = self.api_base
        if self.api_key:
            cfg['api_key'] = self.api_key
        return cfg


class KernelPatchConfig(BaseModel):
    cve_id: str
    origin_repo: str
    target_repo: str
    community_commit_id: str
    target_commit: Optional[str] = None
    artifacts_root: str = 'artifacts'
    debug_mode: bool = True
    max_iterations: int = 2
    prepare_target_worktree: bool = True
    code_query_via_mcp: bool = True
    agentic_retrieval: bool = True
    agentic_retrieval_max_tool_calls: int = 6
    agentic_retrieval_timeout_sec: int = 300
    llm: Optional[KernelPatchLLMConfig] = None

    @field_validator('max_iterations', 'agentic_retrieval_max_tool_calls', 'agentic_retrieval_timeout_sec')
    @classmethod
    def validate_iterations(cls, value: int) -> int:
        if value < 1:
            raise ValueError('value must be >= 1')
        return value

    @property
    def artifact_dir(self) -> Path:
        return Path(self.artifacts_root).expanduser().resolve() / self.cve_id

    @property
    def analysis_path(self) -> Path:
        return self.artifact_dir / f'{self.cve_id}_analysis.md'

    @property
    def patch_path(self) -> Path:
        return self.artifact_dir / f'{self.cve_id}_fix.patch'

    @property
    def trace_path(self) -> Path:
        return self.artifact_dir / f'{self.cve_id}_trace.json'

    @property
    def debug_log_path(self) -> Path:
        return self.artifact_dir / 'agent_debug.log'

    @property
    def worktree_path(self) -> Path:
        return self.artifact_dir / 'target_worktree'

    @property
    def verify_worktree_path(self) -> Path:
        return self.artifact_dir / 'verify_worktree'

    def ensure_artifact_dir(self) -> Path:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.artifact_dir
