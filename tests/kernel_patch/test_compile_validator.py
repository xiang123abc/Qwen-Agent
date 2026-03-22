from qwen_agent.kernel_patch.compile_validator import KernelCompileValidator


def test_compile_validator_skips_without_config(tmp_path):
    repo_root = tmp_path / 'linux-mini'
    repo_root.mkdir()
    (repo_root / 'Makefile').write_text('all:\n\t@true\n', encoding='utf-8')

    validator = KernelCompileValidator(str(repo_root))
    result = validator.validate(repo_root, ['drivers/hid/hid-ntrig.c'])

    assert result.status == 'skipped'
    assert 'No .config found' in result.output
