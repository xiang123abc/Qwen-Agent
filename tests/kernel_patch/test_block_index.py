from qwen_agent.kernel_patch.block_index import build_file_block_index, find_block_by_name, locate_block_by_line


def test_build_file_block_index_detects_top_level_blocks():
    text = """#define PTYPE_HASH_SIZE 16

static const struct seq_operations softnet_seq_ops = {
\t.start = softnet_seq_start,
\t.show = softnet_seq_show,
};

struct ptype_iter_state {
\tint state;
};

static void *ptype_get_idx(struct seq_file *seq, loff_t pos)
{
\treturn NULL;
}
"""
    index = build_file_block_index('net/core/net-procfs.c', text)
    kinds = {(block.kind, block.name) for block in index.blocks}

    assert ('macro', 'PTYPE_HASH_SIZE') in kinds
    assert ('global', 'softnet_seq_ops') in kinds
    assert ('struct', 'ptype_iter_state') in kinds
    assert ('function', 'ptype_get_idx') in kinds

    struct_block = find_block_by_name(index, 'ptype_iter_state')
    assert struct_block is not None
    assert locate_block_by_line(index, struct_block.start_line + 1).name == 'ptype_iter_state'
