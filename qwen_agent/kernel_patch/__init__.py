"""kernel_patch 子包对外导出入口。

该模块聚合了 patch case 读取、仓库操作、edit-unit 解析、评估器与提示词调优等能力，
供 `KernelPatchAgent` 与评测驱动统一引用。
"""

from .case_classifier import classify_case
from .compile_validator import KernelCompileValidator
from .cases import load_patch_cases
from .evaluator import PatchEvaluator
from .edit_units import build_prefetched_context, parse_edit_units, summarize_edit_units
from .autopatch import ClosedLoopPatchPipeline, HeuristicCandidateRetriever, LLMJsonPatchReasoner, LocalContextAssembler
from .autopatch_models import (AutoPatchResult, InsertionAnchor, PatchContext, PatchEdit, PatchPlan,
                               PatchValidationOutcome, RetrievalCandidate, SearchHit, SnippetMatch, VulnerabilityInput)
from .git_ops import KernelRepoManager
from .models import (CaseClassification, CaseRunResult, CodeBlock, CompileValidationResult, EditUnit, FileBlockIndex,
                     KernelCaseBundle, PatchCandidate, PatchCase, PatchEvaluation)
from .prompt_tuner import PromptProfile, load_prompt_profile
from .repo_access import LocalRepoMCPClient

__all__ = [
    'PatchCase',
    'CodeBlock',
    'FileBlockIndex',
    'EditUnit',
    'CaseClassification',
    'CompileValidationResult',
    'KernelCaseBundle',
    'PatchCandidate',
    'PatchEvaluation',
    'CaseRunResult',
    'KernelRepoManager',
    'PatchEvaluator',
    'KernelCompileValidator',
    'VulnerabilityInput',
    'SearchHit',
    'SnippetMatch',
    'InsertionAnchor',
    'RetrievalCandidate',
    'PatchContext',
    'PatchEdit',
    'PatchPlan',
    'PatchValidationOutcome',
    'AutoPatchResult',
    'LocalRepoMCPClient',
    'HeuristicCandidateRetriever',
    'LocalContextAssembler',
    'LLMJsonPatchReasoner',
    'ClosedLoopPatchPipeline',
    'PromptProfile',
    'classify_case',
    'parse_edit_units',
    'summarize_edit_units',
    'build_prefetched_context',
    'load_patch_cases',
    'load_prompt_profile',
]
