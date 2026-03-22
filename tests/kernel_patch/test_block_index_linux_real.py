import importlib.util
import logging
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path('/root/qwen-agent')
PACKAGE_ROOT = REPO_ROOT / 'qwen_agent'
LINUX_REPO = Path('/root/linux')


@lru_cache(maxsize=1)
def _load_kernel_patch_modules():
    qwen_pkg = types.ModuleType('qwen_agent')
    qwen_pkg.__path__ = [str(PACKAGE_ROOT)]
    sys.modules['qwen_agent'] = qwen_pkg

    kernel_patch_pkg = types.ModuleType('qwen_agent.kernel_patch')
    kernel_patch_pkg.__path__ = [str(PACKAGE_ROOT / 'kernel_patch')]
    sys.modules['qwen_agent.kernel_patch'] = kernel_patch_pkg

    log_mod = types.ModuleType('qwen_agent.log')
    log_mod.logger = logging.getLogger('kernel_patch_test')
    sys.modules['qwen_agent.log'] = log_mod

    utils_pkg = types.ModuleType('qwen_agent.utils')
    utils_pkg.__path__ = [str(PACKAGE_ROOT / 'utils')]
    sys.modules['qwen_agent.utils'] = utils_pkg

    utils_mod = types.ModuleType('qwen_agent.utils.utils')
    utils_mod.read_text_from_file = lambda path: Path(path).read_text(encoding='utf-8', errors='replace')
    utils_mod.save_text_to_file = lambda path, text: Path(path).write_text(text, encoding='utf-8')
    sys.modules['qwen_agent.utils.utils'] = utils_mod

    loaded = {}
    for module_name in ('models', 'block_index', 'git_ops'):
        fq_name = f'qwen_agent.kernel_patch.{module_name}'
        spec = importlib.util.spec_from_file_location(fq_name, PACKAGE_ROOT / 'kernel_patch' / f'{module_name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[fq_name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


def _linux_text(relative_path: str, revision: str | None = None) -> str:
    if revision is None:
        return (LINUX_REPO / relative_path).read_text(encoding='utf-8', errors='replace')
    return subprocess.run(['git', '-C', str(LINUX_REPO), 'show', f'{revision}:{relative_path}'],
                          check=True,
                          capture_output=True,
                          text=True,
                          encoding='utf-8').stdout


def _find_line(text: str, needle: str) -> int:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return line_no
    raise AssertionError(f'`{needle}` not found in sample text')


def _write_snapshot(root: Path, relative_path: str, text: str) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def _has_commit(commit: str) -> bool:
    return subprocess.run(['git', '-C', str(LINUX_REPO), 'cat-file', '-e', f'{commit}^{{commit}}']).returncode == 0


pytestmark = pytest.mark.skipif(not (LINUX_REPO / '.git').exists(), reason='requires /root/linux git repository')


def test_build_file_block_index_covers_real_linux_top_level_constructs():
    modules = _load_kernel_patch_modules()
    block_index = modules['block_index']

    process_text = _linux_text('arch/x86/kernel/process.c')
    process_index = block_index.build_file_block_index('arch/x86/kernel/process.c', process_text)

    include_line = _find_line(process_text, '#include <linux/kernel.h>')
    include_block = block_index.locate_block_by_line(process_index, include_line)
    assert (include_block.kind, include_block.name) == ('include', '<linux/kernel.h>')

    global_line = _find_line(process_text, 'DEFINE_PER_CPU_PAGE_ALIGNED(struct tss_struct, cpu_tss_rw)')
    global_block = block_index.locate_block_by_line(process_index, global_line)
    assert (global_block.kind, global_block.name) == ('global', 'cpu_tss_rw')

    variable_line = _find_line(process_text, 'unsigned long boot_option_idle_override = IDLE_NO_OVERRIDE;')
    variable_block = block_index.locate_block_by_line(process_index, variable_line)
    assert (variable_block.kind, variable_block.name) == ('variable', 'boot_option_idle_override')

    processor_text = _linux_text('arch/x86/include/asm/processor.h')
    processor_index = block_index.build_file_block_index('arch/x86/include/asm/processor.h', processor_text)

    macro_line = _find_line(processor_text, '#define X86_VENDOR_INTEL')
    macro_block = block_index.locate_block_by_line(processor_index, macro_line)
    assert (macro_block.kind, macro_block.name) == ('macro', 'X86_VENDOR_INTEL')

    struct_decl_line = _find_line(processor_text, 'struct task_struct;')
    struct_decl_block = block_index.locate_block_by_line(processor_index, struct_decl_line)
    assert (struct_decl_block.kind, struct_decl_block.name) == ('struct_decl', 'task_struct')

    struct_line = _find_line(processor_text, 'struct cpuinfo_x86 {')
    struct_block = block_index.locate_block_by_line(processor_index, struct_line + 1)
    assert (struct_block.kind, struct_block.name) == ('struct', 'cpuinfo_x86')

    fs_text = _linux_text('include/linux/fs.h')
    fs_index = block_index.build_file_block_index('include/linux/fs.h', fs_text)

    extern_var_line = _find_line(fs_text, 'extern unsigned int sysctl_nr_open;')
    extern_var_block = block_index.locate_block_by_line(fs_index, extern_var_line)
    assert (extern_var_block.kind, extern_var_block.name) == ('extern_variable', 'sysctl_nr_open')

    extern_fn_line = _find_line(fs_text, 'extern const char *page_get_link(struct dentry *, struct inode *,')
    extern_fn_block = block_index.locate_block_by_line(fs_index, extern_fn_line)
    assert (extern_fn_block.kind, extern_fn_block.name) == ('extern_function', 'page_get_link')

    boot_text = _linux_text('arch/x86/boot/boot.h')
    boot_index = block_index.build_file_block_index('arch/x86/boot/boot.h', boot_text)

    function_decl_line = _find_line(boot_text, 'void *copy_from_fs(void *dst, addr_t src, size_t len);')
    function_decl_block = block_index.locate_block_by_line(boot_index, function_decl_line)
    assert (function_decl_block.kind, function_decl_block.name) == ('function_decl', 'copy_from_fs')

    idtentry_text = _linux_text('arch/x86/include/asm/idtentry.h')
    idtentry_index = block_index.build_file_block_index('arch/x86/include/asm/idtentry.h', idtentry_text)

    typedef_line = _find_line(idtentry_text, 'typedef void (*idtentry_t)(struct pt_regs *regs);')
    typedef_block = block_index.locate_block_by_line(idtentry_index, typedef_line)
    assert (typedef_block.kind, typedef_block.name) == ('typedef', 'idtentry_t')

    alternative_text = _linux_text('arch/x86/kernel/alternative.c')
    alternative_index = block_index.build_file_block_index('arch/x86/kernel/alternative.c', alternative_text)

    asm_line = _find_line(alternative_text, 'asm(\t".pushsection .rodata')
    asm_block = block_index.locate_block_by_line(alternative_index, asm_line)
    assert asm_block.kind == 'asm'
    assert asm_block.name.startswith('asm@')


@pytest.mark.skipif(not _has_commit('cb3491e875f60580cd984490fd6fec87170d0533'),
                    reason='required Linux commit is unavailable')
@pytest.mark.skipif(not _has_commit('fcfb7ea1f4c62323fb1fe8a893c96a957ba19bea'),
                    reason='required Linux commit is unavailable')
@pytest.mark.skipif(not _has_commit('d5c5afdb9e1efe7e7061e3688356bdae50bfd174'),
                    reason='required Linux commit is unavailable')
def test_kernel_repo_manager_matches_real_linux_commit_snapshots(tmp_path):
    modules = _load_kernel_patch_modules()
    git_ops = modules['git_ops']

    manager = git_ops.KernelRepoManager(str(LINUX_REPO), str(tmp_path / 'workspace'))

    parent_snapshot = tmp_path / 'parent_snapshot'
    rel_path = 'arch/x86/include/asm/processor.h'
    _write_snapshot(parent_snapshot, rel_path, _linux_text(rel_path, 'cb3491e875f60580cd984490fd6fec87170d0533^'))
    struct_block = manager.locate_block(parent_snapshot, rel_path, 84)
    assert (struct_block.kind, struct_block.name) == ('struct', 'cpuinfo_x86')
    index = manager.get_block_index(parent_snapshot, rel_path)
    assert any(block.kind == 'struct' and block.name == 'cpuinfo_x86' for block in index.blocks)

    extern_snapshot = tmp_path / 'extern_snapshot'
    rel_path = 'include/linux/fs.h'
    _write_snapshot(extern_snapshot, rel_path, _linux_text(rel_path, 'fcfb7ea1f4c62323fb1fe8a893c96a957ba19bea^'))
    extern_block = manager.locate_block(extern_snapshot, rel_path, 3088)
    assert (extern_block.kind, extern_block.name, extern_block.start_line, extern_block.end_line) == (
        'extern_function', 'page_get_link', 3088, 3089)

    typedef_snapshot = tmp_path / 'typedef_snapshot'
    rel_path = 'arch/x86/include/asm/idtentry.h'
    _write_snapshot(typedef_snapshot, rel_path, _linux_text(rel_path, 'd5c5afdb9e1efe7e7061e3688356bdae50bfd174'))
    typedef_block = manager.locate_block(typedef_snapshot, rel_path, 16)
    assert (typedef_block.kind, typedef_block.name) == ('typedef', 'idtentry_t')

    before, after = manager.nearest_blocks(typedef_snapshot, rel_path, 17)
    assert (before.kind, before.name) == ('typedef', 'idtentry_t')
    assert (after.kind, after.name) == ('macro', 'DECLARE_IDTENTRY')
