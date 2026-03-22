"""提示词调优器。

根据历史评估结果累计规则得分，把有效的修复启发式动态注入系统提示词。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from .models import PatchEvaluation

RULE_LIBRARY = {
    'strict_patch_format':
        'Always return exactly one unified diff fenced block and nothing else in the patch stage.',
    'use_focused_files_first':
        'Search and patch the community-touched files first; avoid widening scope without strong evidence.',
    'mirror_reference_logic':
        'Replicate the semantic guard or validation logic of the community fix before considering refactors.',
    'minimal_context_preservation':
        'Preserve exact surrounding context lines so the patch applies cleanly with git apply.',
    'preserve_statement_order':
        'Do not reorder existing declarations or statements unless the community patch explicitly does so.',
    'tool_driven_validation':
        'Use search and file-slice tools to verify symbol names, call paths, and nullability before finalizing.',
}


@dataclass
class PromptProfile:
    """可持久化的提示词画像。"""
    version: int = 1
    rule_scores: Dict[str, int] = field(default_factory=dict)
    observations: List[dict] = field(default_factory=list)

    def render_rules(self, max_rules: int = 5) -> List[str]:
        """按分数排序输出最高优先级规则文本。"""
        ranked = sorted(self.rule_scores.items(), key=lambda item: (-item[1], item[0]))
        return [RULE_LIBRARY[key] for key, _ in ranked[:max_rules] if key in RULE_LIBRARY]

    def observe(self, evaluation: PatchEvaluation) -> None:
        """根据一次评估结果更新规则分数与观察记录。"""
        rules = []
        if not evaluation.patch_found:
            rules.extend(['strict_patch_format', 'tool_driven_validation'])
        if evaluation.patch_found and not evaluation.patch_apply_ok:
            rules.extend(['strict_patch_format', 'minimal_context_preservation'])
        if evaluation.patch_apply_ok and not evaluation.tree_match:
            rules.extend(['mirror_reference_logic', 'use_focused_files_first', 'preserve_statement_order'])
        if evaluation.tree_match:
            rules.append('tool_driven_validation')

        # 规则按命中次数累加，形成简单在线学习效果。
        for rule in rules:
            self.rule_scores[rule] = self.rule_scores.get(rule, 0) + 1
        self.observations.append({
            'categories': evaluation.categories,
            'patch_apply_ok': evaluation.patch_apply_ok,
            'tree_match': evaluation.tree_match,
            'similarity': evaluation.similarity,
        })
        self.version += 1

    def save(self, path: Path) -> None:
        """保存 PromptProfile 到 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')


def load_prompt_profile(path: str) -> PromptProfile:
    """从 JSON 文件加载 PromptProfile；不存在时返回默认配置。"""
    profile_path = Path(path)
    if not profile_path.exists():
        return PromptProfile()
    data = json.loads(profile_path.read_text(encoding='utf-8'))
    return PromptProfile(**data)
