import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_agent.agents.closed_loop_kernel_patch_agent import ClosedLoopKernelPatchAgent
from qwen_agent.kernel_patch import KernelRepoManager, VulnerabilityInput, load_patch_cases


RUN_E2E = os.getenv('RUN_KERNEL_PATCH_E2E', '').strip() == '1'
REPO_ROOT = Path('/root/qwen-agent')
LINUX_REPO = Path('/root/linux')
CASES_FILE = REPO_ROOT / 'cve.txt'
EVAL_DRIVER = REPO_ROOT / 'eval_driver.py'


pytestmark = [
    pytest.mark.skipif(not RUN_E2E, reason='set RUN_KERNEL_PATCH_E2E=1 to run model end-to-end test'),
    pytest.mark.skipif(not LINUX_REPO.exists(), reason='requires /root/linux'),
    pytest.mark.skipif(not CASES_FILE.exists(), reason='requires cve.txt'),
]


def _llm_cfg() -> dict:
    api_key = os.getenv('KERNEL_PATCH_E2E_API_KEY') or os.getenv('OPENAI_API_KEY') \
        or ''
    return {
        'model': os.getenv('KERNEL_PATCH_E2E_MODEL', 'qwen3-235b-a22b-thinking-2507'),
        'model_type': 'oai',
        'model_server': os.getenv('KERNEL_PATCH_E2E_API_BASE', 'https://api.apiqik.online/v1'),
        'api_key': api_key,
        'generate_cfg': {
            'temperature': float(os.getenv('KERNEL_PATCH_E2E_TEMPERATURE', '0.1')),
            'max_input_tokens': int(os.getenv('KERNEL_PATCH_E2E_MAX_INPUT_TOKENS', '64000')),
            'max_retries': 1,
        },
    }


def _read_summary(summary_path: Path) -> list[dict]:
    return [json.loads(line) for line in summary_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_eval_driver_runs_one_case_from_cve_txt(tmp_path):
    workspace_root = tmp_path / 'kernel_patch_e2e'
    case_limit = os.getenv('KERNEL_PATCH_E2E_LIMIT', '1').strip() or '1'
    agent_type = os.getenv('KERNEL_PATCH_E2E_AGENT_TYPE', 'kernel_patch').strip() or 'kernel_patch'

    process = subprocess.run(
        [
            sys.executable,
            str(EVAL_DRIVER),
            '--cases-file',
            str(CASES_FILE),
            '--repo-root',
            str(LINUX_REPO),
            '--workspace-root',
            str(workspace_root),
            '--agent-type',
            agent_type,
            '--limit',
            case_limit,
            '--max-attempts',
            os.getenv('KERNEL_PATCH_E2E_ATTEMPTS', '1'),
            '--feedback-style',
            'summary',
            '--keep-worktrees',
            '--recreate-worktrees',
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=os.environ.copy(),
        check=False,
    )

    if process.returncode != 0:
        pytest.fail(f'eval_driver failed with code {process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}')

    summary_path = workspace_root / 'summary.jsonl'
    assert summary_path.exists(), f'summary file not found: {summary_path}'

    records = _read_summary(summary_path)
    assert records, 'summary.jsonl is empty'

    record = records[0]
    assert record['attempts'] >= 1
    assert record['evaluations'], 'evaluation list is empty'
    assert record['evaluations'][0]['patch_found'], f'first attempt did not return a patch: {record}'


def test_closed_loop_kernel_patch_agent_runs_one_case_from_cve_txt(tmp_path):
    cases = load_patch_cases(str(CASES_FILE), limit=1)
    assert cases, 'no valid patch cases loaded from cve.txt'

    manager = KernelRepoManager(str(LINUX_REPO), str(tmp_path / 'closed_loop_workspace'))
    case = cases[0]
    bundle = manager.prepare_case_bundle(case)
    worktree_path = manager.create_worktree(bundle, recreate=True)
    agent = ClosedLoopKernelPatchAgent(llm=_llm_cfg(), repo_manager=manager)

    current_context = manager.current_hunk_context(bundle, worktree_path, context_before=10, context_after=30)
    vulnerability = VulnerabilityInput(
        title=bundle.commit_subject,
        description='\n\n'.join(
            section for section in [bundle.commit_message, bundle.diff_stat, current_context] if section.strip()),
        subsystem_hints=sorted({path.split('/', 1)[0] for path in bundle.changed_files if '/' in path})[:3],
        file_hints=bundle.changed_files[:3],
    )

    try:
        result = agent.run(vulnerability=vulnerability,
                           worktree_path=worktree_path,
                           session_name=f'closed-loop-{case.slug}',
                           max_iterations=1)
    finally:
        manager.remove_worktree(worktree_path)

    assert result.candidate.raw_response.strip(), 'closed-loop agent returned an empty model response'
    assert result.candidate.patch_found, f'closed-loop agent did not emit a patch: {result.candidate.raw_response}'
