import subprocess

from qwen_agent.kernel_patch.evaluator import PatchEvaluator
from qwen_agent.kernel_patch.git_ops import KernelRepoManager
from qwen_agent.kernel_patch.models import PatchCase


def _git(repo_root, *args):
    return subprocess.run(['git', *args],
                          cwd=str(repo_root),
                          check=True,
                          capture_output=True,
                          text=True,
                          encoding='utf-8').stdout.strip()


def test_kernel_repo_manager_worktree_and_evaluation(tmp_path):
    repo_root = tmp_path / 'linux-mini'
    repo_root.mkdir()
    _git(repo_root, 'init')
    _git(repo_root, 'config', 'user.email', 'test@example.com')
    _git(repo_root, 'config', 'user.name', 'Test User')

    target = repo_root / 'drivers' / 'hid'
    target.mkdir(parents=True)
    source_file = target / 'hid-ntrig.c'
    source_file.write_text(
        'int ntrig_report_version(struct hid_device *hdev)\n'
        '{\n'
        '    return hid_to_usb_dev(hdev) ? 0 : -1;\n'
        '}\n',
        encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'base commit')
    base_commit = _git(repo_root, 'rev-parse', 'HEAD')

    source_file.write_text(
        'int ntrig_report_version(struct hid_device *hdev)\n'
        '{\n'
        '    if (!hdev->dev.parent || !hdev->dev.parent->parent)\n'
        '        return -EINVAL;\n'
        '    return hid_to_usb_dev(hdev) ? 0 : -1;\n'
        '}\n',
        encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'fix commit')
    fix_commit = _git(repo_root, 'rev-parse', 'HEAD')

    manager = KernelRepoManager(str(repo_root), str(tmp_path / 'workspace'))
    case = PatchCase(cve_id='CVE-2099-0001', fix_commit=fix_commit, source_line=1)
    bundle = manager.prepare_case_bundle(case)

    assert bundle.base_commit == base_commit
    assert bundle.changed_files == ['drivers/hid/hid-ntrig.c']

    worktree_path = manager.create_worktree(bundle, recreate=True)
    evaluator = PatchEvaluator(manager)
    evaluation = evaluator.evaluate(bundle, worktree_path, bundle.community_patch, bundle.artifact_dir / 'attempt_01')

    assert evaluation.patch_apply_ok
    assert evaluation.tree_match
    assert evaluation.similarity == 1.0


