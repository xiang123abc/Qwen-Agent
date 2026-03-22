"""C/C++ 源文件顶层代码块索引器。

目标是为 patch 定位提供轻量级结构信息：
- 识别 macro / function / inline / struct / union / enum / typedef / global；
- 记录每个块在文件中的起止行号；
- 支持按符号名、行号和邻近关系检索。
"""

import re
from typing import List, Optional, Tuple

from .models import CodeBlock, FileBlockIndex

MACRO_RE = re.compile(r'^\s*#define\s+([A-Za-z_]\w*)\b')
STRUCT_RE = re.compile(r'\b(struct|union|enum)\s+([A-Za-z_]\w*)\b')
TYPEDEF_TAIL_RE = re.compile(r'}\s*([A-Za-z_]\w*)\s*;\s*$')
FUNCTION_RE = re.compile(r'([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:__\w+\s*)*$')
GLOBAL_RE = re.compile(r'([A-Za-z_]\w*)\s*=\s*{\s*$')
CONTROL_KEYWORDS = {'if', 'for', 'while', 'switch'}


def _strip_comments(line: str, in_block_comment: bool) -> Tuple[str, bool]:
    """移除一行中的注释内容，并返回新的 block-comment 状态。"""
    result = []
    i = 0
    while i < len(line):
        if in_block_comment:
            if line[i:i + 2] == '*/':
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if line[i:i + 2] == '/*':
            in_block_comment = True
            i += 2
            continue
        if line[i:i + 2] == '//':
            break
        result.append(line[i])
        i += 1
    return ''.join(result), in_block_comment


def _brace_delta(line: str) -> int:
    """统计一行内 `{` / `}` 的净变化，忽略字符串字面量中的大括号。"""
    delta = 0
    in_string = False
    string_quote = ''
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if in_string:
            if char == string_quote:
                in_string = False
            continue
        if char in ('"', "'"):
            in_string = True
            string_quote = char
            continue
        if char == '{':
            delta += 1
        elif char == '}':
            delta -= 1
    return delta


def _classify_block(path: str, block_lines: List[str], start_line: int, end_line: int) -> Optional[CodeBlock]:
    """将一个带花括号的代码片段归类为可索引的 CodeBlock。"""
    compact = ' '.join(line.strip() for line in block_lines if line.strip())
    if not compact:
        return None

    # 优先识别全局初始化器：`xxx = { ... }`
    header_prefix = compact.split('{', 1)[0].strip() + ' {'
    global_match = GLOBAL_RE.search(header_prefix)
    if global_match:
        return CodeBlock(path=path,
                         kind='global',
                         name=global_match.group(1),
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

    # 再识别函数定义（排除 if/for/while/switch 等控制关键字）。
    function_match = FUNCTION_RE.search(compact.split('{')[0].strip())
    if function_match:
        name = function_match.group(1)
        if name not in CONTROL_KEYWORDS:
            kind = 'function'
            if ' static inline ' in f' {compact} ':
                kind = 'inline'
            return CodeBlock(path=path, kind=kind, name=name, start_line=start_line, end_line=end_line, signature=compact)

    # 最后识别结构体/联合体/枚举以及 typedef 尾命名。
    struct_match = STRUCT_RE.search(compact)
    if struct_match:
        kind = struct_match.group(1)
        name = struct_match.group(2)
        typedef_tail = TYPEDEF_TAIL_RE.search(compact)
        if compact.startswith('typedef ') and typedef_tail:
            kind = 'typedef'
            name = typedef_tail.group(1)
        return CodeBlock(path=path, kind=kind, name=name, start_line=start_line, end_line=end_line, signature=compact)
    return None


def build_file_block_index(path: str, text: str) -> FileBlockIndex:
    """构建单文件顶层代码块索引。"""
    lines = text.splitlines()
    blocks: List[CodeBlock] = []
    i = 0
    in_block_comment = False

    while i < len(lines):
        clean_line, in_block_comment = _strip_comments(lines[i], in_block_comment)
        # 先处理宏定义（含反斜杠续行）。
        macro_match = MACRO_RE.match(clean_line)
        if macro_match:
            start = i + 1
            j = i
            while j + 1 < len(lines) and lines[j].rstrip().endswith('\\'):
                j += 1
            signature = ' '.join(line.strip() for line in lines[i:j + 1])
            blocks.append(CodeBlock(path=path,
                                    kind='macro',
                                    name=macro_match.group(1),
                                    start_line=start,
                                    end_line=j + 1,
                                    signature=signature))
            i = j + 1
            continue

        if not clean_line.strip():
            i += 1
            continue

        # 逐行扩展 header，直到遇到 `{`、`;` 或中断条件。
        header_start = i
        header_lines = [clean_line]
        j = i
        brace_balance = _brace_delta(clean_line)
        saw_open_brace = '{' in clean_line

        while j + 1 < len(lines) and not saw_open_brace:
            next_clean, in_block_comment = _strip_comments(lines[j + 1], in_block_comment)
            if not next_clean.strip():
                break
            if MACRO_RE.match(next_clean):
                break
            header_lines.append(next_clean)
            j += 1
            brace_balance += _brace_delta(next_clean)
            if '{' in next_clean:
                saw_open_brace = True
                break
            if next_clean.strip().endswith(';'):
                break

        if not saw_open_brace:
            i = j + 1
            continue

        # 找到起始 `{` 后继续扫描，直到大括号配平，得到完整顶层块。
        while j + 1 < len(lines) and brace_balance > 0:
            next_clean, in_block_comment = _strip_comments(lines[j + 1], in_block_comment)
            header_lines.append(next_clean)
            j += 1
            brace_balance += _brace_delta(next_clean)

        block = _classify_block(path, lines[header_start:j + 1], header_start + 1, j + 1)
        if block:
            blocks.append(block)
        i = j + 1

    return FileBlockIndex(path=path, blocks=blocks)


def find_block_by_name(index: FileBlockIndex, name: str, kinds: Optional[Tuple[str, ...]] = None) -> Optional[CodeBlock]:
    """按符号名（可选类型过滤）查找块。"""
    normalized = name.strip()
    for block in index.blocks:
        if kinds and block.kind not in kinds:
            continue
        if block.name == normalized:
            return block
    return None


def locate_block_by_line(index: FileBlockIndex, line_no: int) -> Optional[CodeBlock]:
    """按行号定位包含该行的块。"""
    for block in index.blocks:
        if block.start_line <= line_no <= block.end_line:
            return block
    return None


def nearest_blocks(index: FileBlockIndex, line_no: int) -> Tuple[Optional[CodeBlock], Optional[CodeBlock]]:
    """返回目标行前后最近的两个块，用于插入点锚定。"""
    before = None
    after = None
    for block in index.blocks:
        if block.end_line < line_no:
            before = block
            continue
        if block.start_line > line_no:
            after = block
            break
    return before, after
