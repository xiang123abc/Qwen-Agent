import json
import subprocess
from pathlib import Path

from qwen_agent.kernel_patch.config import KernelPatchConfig
from qwen_agent.kernel_patch.mcp_backend import (
    KernelRepoMCPClient,
    _validate_agentic_tool_usage,
    build_target_snippets_via_mcp,
)
from qwen_agent.kernel_patch.models import PatchHunk, RootCauseReport
from qwen_agent.kernel_patch.pipeline import KernelPatchPipeline
from qwen_agent.kernel_patch.prompts import build_agentic_retriever_prompt
from qwen_agent.kernel_patch.runner import KernelPatchDatasetRunner
from qwen_agent.kernel_patch.trace import TraceRecorder
from qwen_agent.llm.schema import ASSISTANT, Message


class SequentialFakeLLM:

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def chat(self, messages, stream=False, **kwargs):
        self.prompts.append('\n'.join(str(message.content) for message in messages))
        if not self.outputs:
            raise AssertionError('No more fake LLM outputs available')
        return [Message(role=ASSISTANT, content=self.outputs.pop(0))]


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(['git', *args], cwd=cwd, text=True, capture_output=True, check=True)
    return proc.stdout.strip()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_test_repos(tmp_path: Path):
    origin = tmp_path / 'origin'
    target = tmp_path / 'target'
    origin.mkdir()
    git(origin, 'init')
    git(origin, 'config', 'user.email', 'test@example.com')
    git(origin, 'config', 'user.name', 'Test User')
    vulnerable = '''#include <string.h>

int vulnerable_copy(const char *src, int len)
{
    char buf[8];

    memcpy(buf, src, len);
    return buf[0];
}
'''
    fixed = '''#include <string.h>

int vulnerable_copy(const char *src, int len)
{
    char buf[8];

    if (len > (int)sizeof(buf))
        return -22;
    memcpy(buf, src, len);
    return buf[0];
}
'''
    source_path = origin / 'drivers' / 'demo.c'
    write(source_path, vulnerable)
    git(origin, 'add', 'drivers/demo.c')
    git(origin, 'commit', '-m', 'demo: add vulnerable copy helper')
    parent_commit = git(origin, 'rev-parse', 'HEAD')
    write(source_path, fixed)
    git(origin, 'commit', '-am', 'demo: fix CVE-TEST overflow')
    fixed_commit = git(origin, 'rev-parse', 'HEAD')
    patch_text = git(origin, 'diff', parent_commit, fixed_commit)
    git(tmp_path, 'clone', str(origin), str(target))
    return origin, target, parent_commit, fixed_commit, patch_text


def build_helper_patch_repos(tmp_path: Path):
    origin = tmp_path / 'origin_helper'
    target = tmp_path / 'target_helper'
    origin.mkdir()
    git(origin, 'init')
    git(origin, 'config', 'user.email', 'test@example.com')
    git(origin, 'config', 'user.name', 'Test User')
    vulnerable = '''#include <string.h>

int vulnerable_copy(const char *src, int len)
{
    char buf[8];

    memcpy(buf, src, len);
    return buf[0];
}
'''
    fixed = '''#include <string.h>

static int checked_len(int len, int max_len)
{
    if (len > max_len)
        return -22;
    return len;
}

int vulnerable_copy(const char *src, int len)
{
    char buf[8];
    int safe_len = checked_len(len, (int)sizeof(buf));

    if (safe_len < 0)
        return safe_len;
    memcpy(buf, src, safe_len);
    return buf[0];
}
'''
    source_path = origin / 'drivers' / 'demo_helper.c'
    write(source_path, vulnerable)
    git(origin, 'add', 'drivers/demo_helper.c')
    git(origin, 'commit', '-m', 'demo: add vulnerable helper case')
    parent_commit = git(origin, 'rev-parse', 'HEAD')
    write(source_path, fixed)
    git(origin, 'commit', '-am', 'demo: fix helper-based CVE overflow')
    fixed_commit = git(origin, 'rev-parse', 'HEAD')
    patch_text = git(origin, 'show', '-W', '--format=medium', '--patch', fixed_commit)
    git(tmp_path, 'clone', str(origin), str(target))
    git(target, 'checkout', parent_commit)
    return origin, target, parent_commit, fixed_commit, patch_text


def decoder_json():
    return json.dumps({
        'vulnerability_type': 'stack overflow',
        'root_cause': 'memcpy copies len bytes into a fixed 8-byte stack buffer without bounds validation.',
        'impacted_files': ['drivers/demo.c'],
        'impacted_functions': ['vulnerable_copy'],
        'impacted_macros': [],
        'impacted_structs': [],
        'impacted_globals': [],
        'fix_logic': ['Validate len against sizeof(buf) before memcpy.'],
        'semantic_anchors': ['vulnerable_copy', 'memcpy(buf, src, len);'],
        'confidence': 'high',
    }, ensure_ascii=False)


def planner_json():
    return json.dumps({
        'summary': 'Patch drivers/demo.c by adding a length check before memcpy.',
        'candidate_files': ['drivers/demo.c'],
        'planned_hunks': [{
            'file_path': 'drivers/demo.c',
            'change_type': 'ADD',
            'rationale': 'The vulnerable_copy implementation still contains the unchecked memcpy call.',
            'anchor': 'memcpy(buf, src, len);',
            'target_snippet': 'drivers/demo.c:5-9',
            'adaptation_strategy': 'Insert an inline guard using sizeof(buf); helper backport not required.'
        }],
        'adaptation_notes': ['Prefer a direct guard because the old tree does not use overflow helpers.'],
        'unresolved_risks': [],
    }, ensure_ascii=False)


def solver_edit_json():
    return json.dumps({
        'summary': 'Insert a bounds check before memcpy.',
        'edits': [{
            'file_path': 'drivers/demo.c',
            'start_line': 6,
            'end_line': 7,
            'new_content': '\n    if (len > (int)sizeof(buf))\n        return -22;\n    memcpy(buf, src, len);\n',
            'rationale': 'Guard the fixed-size buffer before memcpy.'
        }]
    }, ensure_ascii=False)


def failing_patch():
    return '''diff --git a/drivers/demo.c b/drivers/demo.c
--- a/drivers/demo.c
+++ b/drivers/demo.c
@@ -1,3 +1,3 @@
-does not exist
+still does not exist
'''


def test_kernel_patch_pipeline_generates_artifacts(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    debug_calls = []

    def confirm(_config, _plan, iteration):
        debug_calls.append(iteration)
        return True

    pipeline = KernelPatchPipeline(
        llm=SequentialFakeLLM([decoder_json(), planner_json(), patch_text]),
        debug_confirm=confirm,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0001',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
        prepare_target_worktree=True,
    )
    result = pipeline.run(config)

    assert result.success is True
    assert Path(result.analysis_path).exists()
    assert Path(result.patch_path).exists()
    assert Path(result.trace_path).exists()
    assert config.debug_log_path.exists()
    assert debug_calls == [1]
    assert 'git apply --check passed' in result.summary
    trace = json.loads(Path(result.trace_path).read_text(encoding='utf-8'))
    assert trace['success'] is True
    assert trace['decoder_report']['vulnerability_type'] == 'stack overflow'
    debug_log = config.debug_log_path.read_text(encoding='utf-8')
    assert 'solver_snippets_attached' in debug_log
    assert '"event_type": "apply_check"' in debug_log


def test_kernel_patch_pipeline_retries_after_apply_failure(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    fake_llm = SequentialFakeLLM([
        decoder_json(),
        planner_json(),
        failing_patch(),
        planner_json(),
        patch_text,
    ])
    pipeline = KernelPatchPipeline(
        llm=fake_llm,
        debug_confirm=lambda *_args: True,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0002',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=2,
        prepare_target_worktree=True,
    )
    result = pipeline.run(config)

    assert result.success is True
    assert len(result.attempts) == 2
    assert result.attempts[0].apply_check_passed is False
    assert result.attempts[1].apply_check_passed is True
    assert 'previous_apply_error' in fake_llm.prompts[3]


def test_dataset_runner_writes_summary(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    dataset_path = tmp_path / 'cve-use.txt'
    dataset_path.write_text('CVE-TEST-0003,' + fixed_commit + '\n', encoding='utf-8')

    pipeline = KernelPatchPipeline(
        llm=SequentialFakeLLM([decoder_json(), planner_json(), patch_text]),
        debug_confirm=lambda *_args: True,
    )
    runner = KernelPatchDatasetRunner(pipeline=pipeline)
    summary = runner.run_dataset(
        dataset_path=str(dataset_path),
        origin_repo=str(origin),
        target_repo=str(target),
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
    )

    assert summary['total'] == 1
    assert summary['success_count'] == 1
    summary_path = tmp_path / 'artifacts' / 'dataset_summary.json'
    assert summary_path.exists()
    debug_log = (tmp_path / 'artifacts' / 'CVE-TEST-0003' / 'agent_debug.log').read_text(encoding='utf-8')
    assert 'kernel_repo-resolve_parent_commit' in debug_log


def test_coverage_driven_retriever_marks_missing_added_helper(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_helper_patch_repos(tmp_path)
    trace = TraceRecorder(tmp_path / 'retriever_debug.log')
    mcp_client = KernelRepoMCPClient(trace=trace)
    origin_patch = mcp_client.show_commit_patch(str(origin), fixed_commit)
    report = build_target_snippets_via_mcp(
        mcp_client=mcp_client,
        repo=str(target),
        origin_hunks=origin_patch['hunks'],
        semantic_anchors=['vulnerable_copy', 'memcpy(buf, src, safe_len);'],
        path_hints=origin_patch['changed_files'],
        symbol_hints=['vulnerable_copy'],
        origin_patch_text=origin_patch['patch_text'],
    )

    assert report.snippets
    assert any(item.covered for item in report.hunk_coverages)
    assert 'checked_len' in report.added_functions
    assert 'checked_len' in report.missing_entities
    assert any('demo_helper.c' in snippet.file_path for snippet in report.snippets)


def test_planner_prompt_includes_retrieval_report(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    fake_llm = SequentialFakeLLM([decoder_json(), planner_json(), patch_text])
    pipeline = KernelPatchPipeline(
        llm=fake_llm,
        debug_confirm=lambda *_args: True,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0004',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
        prepare_target_worktree=True,
        code_query_via_mcp=True,
    )
    pipeline.run(config)

    planner_prompt = fake_llm.prompts[1]
    assert 'retrieval_report' in planner_prompt
    assert 'missing_entities' in planner_prompt
    assert 'hunk_coverages' in planner_prompt


def test_solver_prompt_includes_retrieval_report(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    fake_llm = SequentialFakeLLM([decoder_json(), planner_json(), patch_text])
    pipeline = KernelPatchPipeline(
        llm=fake_llm,
        debug_confirm=lambda *_args: True,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0005',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
        prepare_target_worktree=True,
        code_query_via_mcp=True,
    )
    pipeline.run(config)

    solver_prompt = fake_llm.prompts[2]
    assert 'retrieval_report' in solver_prompt
    assert 'missing_entities' in solver_prompt
    assert 'hunk_coverages' in solver_prompt
    assert 'solver_snippets' in solver_prompt
    assert 'edits' in solver_prompt


def test_analysis_markdown_includes_retrieval_summary(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    fake_llm = SequentialFakeLLM([decoder_json(), planner_json(), patch_text])
    pipeline = KernelPatchPipeline(
        llm=fake_llm,
        debug_confirm=lambda *_args: True,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0006',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
        prepare_target_worktree=True,
        code_query_via_mcp=True,
    )
    pipeline.run(config)

    analysis = Path(config.analysis_path).read_text(encoding='utf-8')
    assert '## Retriever' in analysis
    assert 'Covered Hunks:' in analysis
    assert 'Missing Entities:' in analysis
    assert '### Hunk Coverage' in analysis


def test_solver_snippets_are_attached_to_fix_plan(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    fake_llm = SequentialFakeLLM([decoder_json(), planner_json(), patch_text])
    pipeline = KernelPatchPipeline(
        llm=fake_llm,
        debug_confirm=lambda *_args: True,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0007',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
        prepare_target_worktree=True,
        code_query_via_mcp=True,
    )
    result = pipeline.run(config)
    assert result.final_plan.solver_snippets
    assert any(snippet.file_path == 'drivers/demo.c' for snippet in result.final_plan.solver_snippets)


def test_solver_edit_plan_materializes_patch(tmp_path):
    origin, target, parent_commit, fixed_commit, patch_text = build_test_repos(tmp_path)
    fake_llm = SequentialFakeLLM([decoder_json(), planner_json(), solver_edit_json()])
    pipeline = KernelPatchPipeline(
        llm=fake_llm,
        debug_confirm=lambda *_args: True,
    )
    config = KernelPatchConfig(
        cve_id='CVE-TEST-0008',
        origin_repo=str(origin),
        target_repo=str(target),
        community_commit_id=fixed_commit,
        target_commit=parent_commit,
        artifacts_root=str(tmp_path / 'artifacts'),
        debug_mode=True,
        max_iterations=1,
        prepare_target_worktree=True,
        code_query_via_mcp=True,
    )
    result = pipeline.run(config)
    assert result.success is True
    patch_text = Path(result.patch_path).read_text(encoding='utf-8')
    assert 'if (len > (int)sizeof(buf))' in patch_text


def test_agentic_retriever_prompt_prioritizes_changed_files_and_impacted_functions():
    prompt = build_agentic_retriever_prompt(
        decoder_report=RootCauseReport(
            vulnerability_type='uaf',
            root_cause='test',
            impacted_files=['drivers/demo.c'],
            impacted_functions=['demo_fn'],
        ),
        origin_hunks=[PatchHunk(file_path='drivers/demo.c', old_start=10, old_count=3, new_start=10, new_count=5)],
        changed_files=['drivers/demo.c'],
        origin_patch='diff --git a/drivers/demo.c b/drivers/demo.c',
        repo='/tmp/repo',
    )
    assert '前 2 次工具调用必须优先围绕 changed_files 和 impacted_functions' in prompt
    assert '第 1 优先动作' in prompt
    assert '只有在 changed_files 和 impacted_functions 已经查过之后' in prompt


def test_agentic_retriever_rejects_regex_like_search_code():
    responses = [
        Message(
            role=ASSISTANT,
            content='',
            function_call={'name': 'kernel_repo-search_code', 'arguments': json.dumps({
                'repo': '/tmp/repo',
                'pattern': 'static void \\*ptype_get_idx',
                'path_glob': 'net/core/net-procfs.c',
            })}
        )
    ]
    try:
        _validate_agentic_tool_usage(responses, ['net/core/net-procfs.c'], 'kernel_repo')
    except RuntimeError as ex:
        assert 'regex-like pattern' in str(ex)
    else:
        raise AssertionError('Expected regex-like search_code pattern to be rejected')
