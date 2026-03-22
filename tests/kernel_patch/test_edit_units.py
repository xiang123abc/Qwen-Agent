import subprocess

from qwen_agent.kernel_patch.case_classifier import classify_case
from qwen_agent.kernel_patch.edit_units import parse_edit_units
from qwen_agent.kernel_patch.git_ops import KernelRepoManager
from qwen_agent.kernel_patch.models import PatchCase


def _git(repo_root, *args):
    return subprocess.run(['git', *args],
                          cwd=str(repo_root),
                          check=True,
                          capture_output=True,
                          text=True,
                          encoding='utf-8').stdout.strip()


def test_parse_edit_units_handles_added_struct_and_multi_function_patch(tmp_path):
    repo_root = tmp_path / 'linux-mini'
    repo_root.mkdir()
    _git(repo_root, 'init')
    _git(repo_root, 'config', 'user.email', 'test@example.com')
    _git(repo_root, 'config', 'user.name', 'Test User')

    target = repo_root / 'net' / 'core'
    target.mkdir(parents=True)
    source_file = target / 'net-procfs.c'
    source_file.write_text(
        'static const struct seq_operations softnet_seq_ops = {\n'
        '\t.show = softnet_seq_show,\n'
        '};\n\n'
        'static void *ptype_get_idx(struct seq_file *seq, loff_t pos)\n'
        '{\n\treturn NULL;\n}\n\n'
        'static void *ptype_seq_next(struct seq_file *seq, void *v, loff_t *pos)\n'
        '{\n\treturn NULL;\n}\n\n'
        'static int __net_init dev_proc_net_init(struct net *net)\n'
        '{\n\treturn proc_create_net("ptype", 0444, net->proc_net, &ptype_seq_ops,\n'
        '\t\t\tsizeof(struct seq_net_private)) ? 0 : -ENOMEM;\n}\n',
        encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'base commit')

    source_file.write_text(
        'static const struct seq_operations softnet_seq_ops = {\n'
        '\t.show = softnet_seq_show,\n'
        '};\n\n'
        'struct ptype_iter_state {\n\tstruct seq_net_private p;\n\tstruct net_device *dev;\n};\n\n'
        'static void *ptype_get_idx(struct seq_file *seq, loff_t pos)\n'
        '{\n\tstruct ptype_iter_state *iter = seq->private;\n\titer->dev = NULL;\n\treturn NULL;\n}\n\n'
        'static void *ptype_seq_next(struct seq_file *seq, void *v, loff_t *pos)\n'
        '{\n\tstruct ptype_iter_state *iter = seq->private;\n\treturn NULL;\n}\n\n'
        'static int __net_init dev_proc_net_init(struct net *net)\n'
        '{\n\treturn proc_create_net("ptype", 0444, net->proc_net, &ptype_seq_ops,\n'
        '\t\t\tsizeof(struct ptype_iter_state)) ? 0 : -ENOMEM;\n}\n',
        encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'net: add proper RCU protection')
    fix_commit = _git(repo_root, 'rev-parse', 'HEAD')

    manager = KernelRepoManager(str(repo_root), str(tmp_path / 'workspace'))
    bundle = manager.prepare_case_bundle(PatchCase(cve_id='CVE-2099-0002', fix_commit=fix_commit, source_line=1))
    worktree_path = manager.create_worktree(bundle, recreate=True)
    units = parse_edit_units(bundle, manager, worktree_path)

    assert len(units) >= 3
    assert any(unit.operation == 'add_block' and unit.block_kind == 'struct' for unit in units)
    assert any(unit.symbol == 'ptype_get_idx' for unit in units)
    assert any(unit.symbol == 'dev_proc_net_init' for unit in units)

    classification = classify_case(bundle, units)
    assert classification.primary_kind in {'block_structure_patch', 'coordinated_logic_patch'}
    assert 'adds_block' in classification.labels
