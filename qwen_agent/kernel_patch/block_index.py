"""C/C++ 源文件顶层代码块索引器。

目标是为 patch 定位提供轻量级结构信息：
- 识别 include / macro / function / inline / struct / union / enum / typedef；
- 识别 extern / variable / asm / 前向声明 / 函数声明等顶层可锚定声明；
- 记录每个块在文件中的起止行号；
- 支持按符号名、行号和邻近关系检索。
"""

import re
from typing import List, Optional, Tuple

from .models import CodeBlock, FileBlockIndex

INCLUDE_RE = re.compile(r'^\s*#include\s+([<"][^>"]+[>"])')
MACRO_RE = re.compile(r'^\s*#define\s+([A-Za-z_]\w*)\b')
STRUCT_RE = re.compile(r'\b(struct|union|enum)\s+([A-Za-z_]\w*)\b')
TYPEDEF_TAIL_RE = re.compile(r'}\s*([A-Za-z_]\w*)\s*;\s*$')
FUNCTION_RE = re.compile(r'([A-Za-z_]\w*)\s*\((?:[^()]|\([^()]*\))*\)\s*(?:__\w+(?:\s*\([^)]*\))?\s*)*$')
TYPEDEF_FN_PTR_RE = re.compile(r'\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(')
TYPEDEF_FN_TYPE_RE = re.compile(r'\(\s*([A-Za-z_]\w*)\s*\)\s*\(')
TYPEDEF_MACRO_RE = re.compile(r'^\s*typedef\s+[A-Za-z_]\w*\s*\(\s*([A-Za-z_]\w*)\b')
FORWARD_DECL_RE = re.compile(r'^\s*(struct|union|enum)\s+([A-Za-z_]\w*)\s*;\s*$')
ASM_RE = re.compile(r'^\s*(?:asm|__asm__)\b')
EXTERN_RE = re.compile(r'^\s*extern\b')
DECL_ATTRIBUTE_RE = re.compile(r'(?:\s+(?:__attribute__\s*\(\([^;]*\)\)|__(?:\w+)(?:\s*\([^;]*\))?))+\s*$')
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


def _is_preprocessor_line(line: str) -> bool:
    return line.lstrip().startswith('#')


def _scan_syntax(line: str) -> Tuple[int, int, bool, bool]:
    """扫描一行中的括号/分号信息，忽略字符串字面量内的符号。"""
    brace_delta = 0
    paren_delta = 0
    saw_open_brace = False
    has_semicolon = False
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
            brace_delta += 1
            saw_open_brace = True
        elif char == '}':
            brace_delta -= 1
        elif char == '(':
            paren_delta += 1
        elif char == ')':
            paren_delta -= 1
        elif char == ';':
            has_semicolon = True
    return brace_delta, paren_delta, saw_open_brace, has_semicolon


def _last_identifier(text: str) -> Optional[str]:
    matches = list(re.finditer(r'[A-Za-z_]\w*', text))
    return matches[-1].group(0) if matches else None


def _last_identifier_outside_groups(text: str) -> Optional[str]:
    identifiers = []
    in_string = False
    string_quote = ''
    escaped = False
    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0
    current = []

    def flush_current():
        nonlocal current
        if current and paren_depth == 0 and brace_depth == 0 and bracket_depth == 0:
            identifiers.append(''.join(current))
        current = []

    for char in text:
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
            flush_current()
            in_string = True
            string_quote = char
            continue
        if char == '(':
            flush_current()
            paren_depth += 1
            continue
        if char == ')':
            flush_current()
            paren_depth = max(0, paren_depth - 1)
            continue
        if char == '{':
            flush_current()
            brace_depth += 1
            continue
        if char == '}':
            flush_current()
            brace_depth = max(0, brace_depth - 1)
            continue
        if char == '[':
            flush_current()
            bracket_depth += 1
            continue
        if char == ']':
            flush_current()
            bracket_depth = max(0, bracket_depth - 1)
            continue
        if char == '_' or char.isalnum():
            current.append(char)
            continue
        flush_current()
    flush_current()
    return identifiers[-1] if identifiers else None


def _strip_trailing_decl_attributes(text: str) -> str:
    stripped = text.rstrip()
    while True:
        updated = DECL_ATTRIBUTE_RE.sub('', stripped)
        if updated == stripped:
            return stripped
        stripped = updated.rstrip()


def _extract_assignment_name(lhs: str) -> Optional[str]:
    stripped = _strip_trailing_decl_attributes(lhs.strip())
    if stripped.endswith(')'):
        depth = 0
        start = None
        for index in range(len(stripped) - 1, -1, -1):
            char = stripped[index]
            if char == ')':
                depth += 1
            elif char == '(':
                depth -= 1
                if depth == 0:
                    start = index
                    break
        if start is not None:
            arg_name = _last_identifier_outside_groups(stripped[start + 1:-1])
            if arg_name:
                return arg_name
    return _last_identifier_outside_groups(stripped)


def _function_name(signature: str) -> Optional[str]:
    function_match = FUNCTION_RE.search(signature.strip())
    if not function_match:
        return None
    name = function_match.group(1)
    if name in CONTROL_KEYWORDS:
        return None
    return name


def _classify_braced_block(path: str, block_lines: List[str], start_line: int, end_line: int) -> Optional[CodeBlock]:
    """将一个带花括号的代码片段归类为可索引的 CodeBlock。"""
    compact = ' '.join(line.strip() for line in block_lines if line.strip())
    if not compact:
        return None

    # 优先识别全局初始化器：`xxx = { ... }`
    header_prefix = compact.split('{', 1)[0].strip()
    global_name = _extract_assignment_name(header_prefix.split('=', 1)[0]) if '=' in header_prefix else None
    if global_name:
        return CodeBlock(path=path,
                         kind='global',
                         name=global_name,
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

    # 再识别函数定义（排除 if/for/while/switch 等控制关键字）。
    function_name = _function_name(compact.split('{', 1)[0].strip())
    if function_name:
        kind = 'function'
        if ' inline ' in f' {compact} ':
            kind = 'inline'
        return CodeBlock(path=path,
                         kind=kind,
                         name=function_name,
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

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


def _classify_statement_block(path: str, block_lines: List[str], start_line: int, end_line: int) -> Optional[CodeBlock]:
    """将一个顶层 `;` 结束的声明/语句归类为可索引的 CodeBlock。"""
    compact = ' '.join(line.strip() for line in block_lines if line.strip())
    if not compact:
        return None

    include_match = INCLUDE_RE.match(compact)
    if include_match:
        return CodeBlock(path=path,
                         kind='include',
                         name=include_match.group(1),
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

    if ASM_RE.match(compact):
        return CodeBlock(path=path,
                         kind='asm',
                         name=f'asm@{start_line}',
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

    if compact.startswith('typedef '):
        typedef_match = TYPEDEF_FN_PTR_RE.search(compact)
        if typedef_match:
            return CodeBlock(path=path,
                             kind='typedef',
                             name=typedef_match.group(1),
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        typedef_function_type = TYPEDEF_FN_TYPE_RE.search(compact)
        if typedef_function_type:
            return CodeBlock(path=path,
                             kind='typedef',
                             name=typedef_function_type.group(1),
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        typedef_tail = TYPEDEF_TAIL_RE.search(compact)
        if typedef_tail:
            return CodeBlock(path=path,
                             kind='typedef',
                             name=typedef_tail.group(1),
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        typedef_macro = TYPEDEF_MACRO_RE.match(compact)
        if typedef_macro:
            return CodeBlock(path=path,
                             kind='typedef',
                             name=typedef_macro.group(1),
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        alias_name = _last_identifier(_strip_trailing_decl_attributes(compact[:-1]))
        if alias_name and alias_name != 'typedef':
            return CodeBlock(path=path,
                             kind='typedef',
                             name=alias_name,
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        return None

    forward_decl = FORWARD_DECL_RE.match(compact)
    if forward_decl:
        return CodeBlock(path=path,
                         kind=f'{forward_decl.group(1)}_decl',
                         name=forward_decl.group(2),
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

    if EXTERN_RE.match(compact):
        extern_tail = compact[len('extern '):].strip()
        function_name = _function_name(extern_tail.rstrip(';'))
        if function_name:
            return CodeBlock(path=path,
                             kind='extern_function',
                             name=function_name,
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        variable_name = _last_identifier_outside_groups(_strip_trailing_decl_attributes(extern_tail[:-1]))
        if variable_name:
            return CodeBlock(path=path,
                             kind='extern_variable',
                             name=variable_name,
                             start_line=start_line,
                             end_line=end_line,
                             signature=compact)
        return None

    function_name = _function_name(compact.rstrip(';'))
    if function_name:
        return CodeBlock(path=path,
                         kind='function_decl',
                         name=function_name,
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)

    if '=' in compact:
        variable_name = _extract_assignment_name(compact[:-1].split('=', 1)[0])
    else:
        variable_name = _last_identifier_outside_groups(_strip_trailing_decl_attributes(compact[:-1]))
    if variable_name:
        return CodeBlock(path=path,
                         kind='variable',
                         name=variable_name,
                         start_line=start_line,
                         end_line=end_line,
                         signature=compact)
    return None


def build_file_block_index(path: str, text: str) -> FileBlockIndex:
    """构建单文件顶层代码块索引。"""
    lines = text.splitlines()
    blocks: List[CodeBlock] = []
    i = 0
    in_block_comment = False

    while i < len(lines):
        clean_line, in_block_comment = _strip_comments(lines[i], in_block_comment)
        include_match = INCLUDE_RE.match(clean_line)
        if include_match:
            blocks.append(CodeBlock(path=path,
                                    kind='include',
                                    name=include_match.group(1),
                                    start_line=i + 1,
                                    end_line=i + 1,
                                    signature=clean_line.strip()))
            i += 1
            continue

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

        if _is_preprocessor_line(clean_line):
            i += 1
            continue

        if not clean_line.strip():
            i += 1
            continue

        # 逐行扩展 header，直到遇到 `{`、`;` 或中断条件。
        header_start = i
        header_lines = [clean_line]
        j = i
        brace_balance, paren_balance, saw_open_brace, has_semicolon = _scan_syntax(clean_line)
        statement_complete = has_semicolon and brace_balance == 0 and paren_balance == 0

        while j + 1 < len(lines) and not saw_open_brace and not statement_complete:
            next_clean, in_block_comment = _strip_comments(lines[j + 1], in_block_comment)
            if _is_preprocessor_line(next_clean):
                break
            if MACRO_RE.match(next_clean):
                break
            if INCLUDE_RE.match(next_clean):
                break
            if not next_clean.strip() and brace_balance == 0 and paren_balance == 0:
                break
            header_lines.append(next_clean)
            j += 1
            line_braces, line_parens, line_has_open_brace, line_has_semicolon = _scan_syntax(next_clean)
            brace_balance += line_braces
            paren_balance += line_parens
            saw_open_brace = saw_open_brace or line_has_open_brace
            statement_complete = line_has_semicolon and brace_balance == 0 and paren_balance == 0

        if saw_open_brace:
            # 找到起始 `{` 后继续扫描，直到大括号配平，得到完整顶层块。
            while j + 1 < len(lines) and brace_balance > 0:
                next_clean, in_block_comment = _strip_comments(lines[j + 1], in_block_comment)
                header_lines.append(next_clean)
                j += 1
                line_braces, _, _, _ = _scan_syntax(next_clean)
                brace_balance += line_braces
            block = _classify_braced_block(path, lines[header_start:j + 1], header_start + 1, j + 1)
        elif statement_complete:
            block = _classify_statement_block(path, lines[header_start:j + 1], header_start + 1, j + 1)
        else:
            block = None

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
