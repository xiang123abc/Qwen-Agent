import subprocess

from qwen_agent.kernel_patch.git_ops import KernelRepoManager
from qwen_agent.kernel_patch.repo_access import LocalRepoMCPClient


def _git(repo_root, *args):
    return subprocess.run(['git', *args],
                          cwd=str(repo_root),
                          check=True,
                          capture_output=True,
                          text=True,
                          encoding='utf-8').stdout.strip()


def test_local_repo_mcp_client_locates_exact_and_fuzzy_snippets(tmp_path):
    repo_root = tmp_path / 'linux-mini'
    repo_root.mkdir()
    _git(repo_root, 'init')
    _git(repo_root, 'config', 'user.email', 'test@example.com')
    _git(repo_root, 'config', 'user.name', 'Test User')

    target = repo_root / 'drivers' / 'misc'
    target.mkdir(parents=True)
    source_file = target / 'sample.c'
    source_file.write_text(
        'int foo(struct ctx *ctx)\n'
        '{\n'
        '    if (!ctx)\n'
        '        return -EINVAL;\n'
        '\n'
        '    do_work(ctx);\n'
        '    return 0;\n'
        '}\n'
        '\n'
        'static int bar(void)\n'
        '{\n'
        '    return foo(NULL);\n'
        '}\n',
        encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'base commit')

    manager = KernelRepoManager(str(repo_root), str(tmp_path / 'workspace'))
    client = LocalRepoMCPClient(manager)

    exact = client.locate_snippet(repo_root,
                                  'drivers/misc/sample.c',
                                  'if (!ctx)\n'
                                  '    return -EINVAL;\n')
    assert exact
    assert exact[0].strategy == 'exact'
    assert (exact[0].start_line, exact[0].end_line) == (3, 4)

    fuzzy = client.locate_snippet(
        repo_root,
        'drivers/misc/sample.c',
        'if (!ctx) {\n'
        '    return -EINVAL;\n'
        '}\n',
    )
    assert fuzzy
    assert fuzzy[0].strategy == 'fuzzy'
    assert fuzzy[0].score >= 0.60
    assert fuzzy[0].start_line == 3

    anchor = client.resolve_insertion_point(repo_root,
                                            'drivers/misc/sample.c',
                                            before_snippet='if (!ctx)\n    return -EINVAL;\n',
                                            after_snippet='do_work(ctx);\n')
    assert anchor is not None
    assert anchor.strategy == 'between_anchors'
    assert anchor.line_no == 5


def test_local_repo_mcp_client_reads_revision_text(tmp_path):
    repo_root = tmp_path / 'linux-mini'
    repo_root.mkdir()
    _git(repo_root, 'init')
    _git(repo_root, 'config', 'user.email', 'test@example.com')
    _git(repo_root, 'config', 'user.name', 'Test User')

    target = repo_root / 'kernel'
    target.mkdir(parents=True)
    source_file = target / 'demo.c'
    source_file.write_text('int demo(void)\n{\n    return 0;\n}\n', encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'base commit')
    base_commit = _git(repo_root, 'rev-parse', 'HEAD')

    source_file.write_text('int demo(void)\n{\n    return 1;\n}\n', encoding='utf-8')
    _git(repo_root, 'add', '.')
    _git(repo_root, 'commit', '-m', 'update demo')

    manager = KernelRepoManager(str(repo_root), str(tmp_path / 'workspace'))
    client = LocalRepoMCPClient(manager)
    old_text = client.read_revision_text(base_commit, 'kernel/demo.c')

    assert 'return 0;' in old_text
    assert 'return 1;' not in old_text
