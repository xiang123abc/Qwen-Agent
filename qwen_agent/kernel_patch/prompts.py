from typing import Iterable

from .models import KernelCaseBundle

BASE_SYSTEM_PROMPT = """You are a Linux kernel vulnerability repair agent built on qwen-agent.

Your job is to reproduce a high-quality fix patch against the provided parent commit worktree.

Mandatory rules:
1. Analyze root cause before proposing any patch.
2. Start from the files touched by the community fix. Expand search scope only when evidence requires it.
3. Keep the fix minimal and kernel-style correct. Do not make unrelated cleanups.
4. Use repository tools to inspect code before finalizing.
5. The final patch must be a valid unified diff from repository root.
6. When you output the final patch, place it inside one fenced ```diff block.
7. Preserve existing statement order, indentation, and tab style unless the community patch explicitly changes them.
"""

ANALYSIS_PROMPT_TEMPLATE = """You are solving a Linux kernel vulnerability backport case.

Case:
- CVE: {cve_id}
- Base commit (worktree HEAD): {base_commit}
- Community fix commit: {fix_commit}
- Commit subject: {commit_subject}

Focused files from the community fix:
{changed_files}

Community diff stat:
{diff_stat}

Reference material from `git show -W {fix_commit}`:
{reference_show_excerpt}

Task:
1. Explain the vulnerability trigger and root cause.
2. Compare the pre-fix and post-fix logic.
3. Identify the minimal files and code regions that must change in the parent worktree.
4. Produce a concise repair plan.

Community patch excerpt:
{community_patch_excerpt}

Current parent-worktree excerpts matching the community hunk locations:
{current_context_excerpt}

Output format:
# Root Cause
# Before Fix
# After Fix
# Repair Plan
# Files To Edit

Do not output a patch in this stage.
If the commit message and the community patch disagree, trust the actual patch hunk.
"""

PATCH_PROMPT_TEMPLATE = """Use the root-cause analysis below to generate the final patch against the current parent-commit worktree.

Analysis:
{analysis_text}

Additional retry feedback:
{feedback_text}

Current parent-worktree excerpts:
{current_context_excerpt}

Requirements:
- Output exactly one unified diff patch in a ```diff fenced block.
- The patch must apply with `git apply`.
- Prefer `git diff -U3` style hunks with stable unchanged context.
- Do not invent `index` lines or blob hashes.
- Do not include explanations outside the patch block.
- Prefer the focused files first and keep the patch minimal.
- Before emitting the patch, inspect the current target function or symbol in the parent worktree.
- If the community patch and the commit message conflict, follow the community patch.
- Preserve the surrounding indentation exactly; do not replace kernel tabs with spaces in touched lines.
"""


def render_system_prompt(extra_rules: Iterable[str]) -> str:
    extra = [rule.strip() for rule in extra_rules if rule and rule.strip()]
    if not extra:
        return BASE_SYSTEM_PROMPT
    return BASE_SYSTEM_PROMPT + '\nLearned repair heuristics:\n' + '\n'.join(f'- {rule}' for rule in extra)


def build_analysis_prompt(bundle: KernelCaseBundle, current_context_excerpt: str) -> str:
    changed_files = '\n'.join(f'- {path}' for path in bundle.changed_files) or '- (no changed files reported)'
    return ANALYSIS_PROMPT_TEMPLATE.format(cve_id=bundle.case.cve_id,
                                           base_commit=bundle.base_commit,
                                           fix_commit=bundle.fix_commit,
                                           commit_subject=bundle.commit_subject,
                                           changed_files=changed_files,
                                           diff_stat=bundle.diff_stat or '(empty)',
                                           reference_show_excerpt=bundle.reference_show_excerpt.strip() or '(empty)',
                                           community_patch_excerpt=bundle.community_patch_excerpt.strip() or '(empty)',
                                           current_context_excerpt=current_context_excerpt.strip() or '(empty)')


def build_patch_prompt(analysis_text: str, current_context_excerpt: str, feedback_text: str = '') -> str:
    return PATCH_PROMPT_TEMPLATE.format(analysis_text=analysis_text.strip() or '(analysis unavailable)',
                                        feedback_text=feedback_text.strip() or '(none)',
                                        current_context_excerpt=current_context_excerpt.strip() or '(empty)')
