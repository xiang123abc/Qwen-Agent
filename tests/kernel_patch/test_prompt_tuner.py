from qwen_agent.kernel_patch.models import PatchEvaluation
from qwen_agent.kernel_patch.prompt_tuner import PromptProfile


def test_prompt_profile_learns_from_failed_patch():
    profile = PromptProfile()
    evaluation = PatchEvaluation(
        patch_found=True,
        patch_apply_ok=False,
        tree_match=False,
        similarity=0.0,
        categories=['patch_apply_failed'],
    )

    profile.observe(evaluation)
    rules = profile.render_rules()

    assert rules
    assert any('unified diff fenced block' in rule for rule in rules)
