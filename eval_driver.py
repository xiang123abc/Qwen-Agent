import argparse
import json
import os
from pathlib import Path
from typing import List

from qwen_agent.agents.kernel_patch_agent import KernelPatchAgent
from qwen_agent.kernel_patch import CaseRunResult, KernelRepoManager, PatchEvaluator, load_patch_cases, load_prompt_profile
from qwen_agent.kernel_patch.compile_validator import KernelCompileValidator
from qwen_agent.log import configure_logger, logger


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end Linux CVE patch generation driver based on qwen-agent.')
    parser.add_argument('--cases-file', default='cve.txt')
    parser.add_argument('--repo-root', default='/root/linux')
    parser.add_argument('--workspace-root', default='workspace/kernel_patch_eval')
    parser.add_argument('--model', default='qwen3-235b-a22b-thinking-2507')
    parser.add_argument('--api-base', default='https://api.apiqik.online/v1')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--cve-id', default='')
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--max-input-tokens', type=int, default=64000)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--feedback-style', choices=['summary', 'full_diff'], default='summary')
    parser.add_argument('--compile-validation', action='store_true')
    parser.add_argument('--log-level', default='')
    parser.add_argument('--log-file', default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--keep-worktrees', action='store_true')
    parser.add_argument('--recreate-worktrees', action='store_true')
    return parser.parse_args()


def build_llm_cfg(args) -> dict:
    api_key = args.api_key or os.getenv(args.api_key_env, '')
    if not api_key:
        raise SystemExit(f'API key is required. Provide --api-key or set {args.api_key_env}.')
    return {
        'model': args.model,
        'model_type': 'oai',
        'model_server': args.api_base,
        'api_key': api_key,
        'generate_cfg': {
            'temperature': args.temperature,
            'max_input_tokens': args.max_input_tokens,
            'max_retries': 1,
        }
    }


def save_attempt_files(attempt_dir: Path, candidate, evaluation) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / 'analysis.md').write_text(candidate.analysis_text or '', encoding='utf-8')
    (attempt_dir / 'raw_response.md').write_text(candidate.raw_response or '', encoding='utf-8')
    if candidate.patch_text:
        (attempt_dir / 'model.patch').write_text(candidate.patch_text, encoding='utf-8')
    (attempt_dir / 'evaluation.json').write_text(json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2),
                                                 encoding='utf-8')


def choose_best_result(results: List) -> float:
    best = 0.0
    for evaluation in results:
        if evaluation.tree_match:
            return 1.0
        best = max(best, evaluation.similarity)
    return best


def select_learning_evaluation(results: List):
    tree_matches = [item for item in results if item.tree_match]
    if tree_matches:
        return tree_matches[0]

    applicable = [item for item in results if item.patch_apply_ok]
    if applicable:
        return max(applicable, key=lambda item: item.similarity)

    return results[-1]


def main():
    args = parse_args()
    log_level = args.log_level or ('DEBUG' if args.debug else None)
    # configure_logger(level=log_level, log_file=args.log_file or None)
    workspace_root = Path(args.workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    prompt_profile_path = workspace_root / 'prompt_profile.json'
    prompt_profile = load_prompt_profile(str(prompt_profile_path))
    llm_cfg = build_llm_cfg(args)
    repo_manager = KernelRepoManager(repo_root=args.repo_root, workspace_root=str(workspace_root))
    compile_validator = KernelCompileValidator(args.repo_root) if args.compile_validation else None
    evaluator = PatchEvaluator(repo_manager,
                               feedback_style=args.feedback_style,
                               compile_validator=compile_validator)
    agent = KernelPatchAgent(llm=llm_cfg, repo_manager=repo_manager, prompt_profile=prompt_profile)

    cases = load_patch_cases(args.cases_file, cve_filter=args.cve_id or None, limit=args.limit)
    if not cases:
        raise SystemExit('No valid CVE cases found.')

    summary_path = workspace_root / 'summary.jsonl'
    if summary_path.exists():
        summary_path.unlink()

    for case in cases:
        logger.info(f'Running case {case.cve_id} -> {case.fix_commit}')
        bundle = repo_manager.prepare_case_bundle(case)
        worktree_path = repo_manager.create_worktree(bundle, recreate=args.recreate_worktrees)
        case_results = []
        feedback_text = ''
        try:
            for attempt in range(1, args.max_attempts + 1):
                attempt_dir = bundle.artifact_dir / f'attempt_{attempt:02d}'
                repo_manager.reset_worktree(worktree_path, bundle.base_commit)
                candidate = agent.generate_candidate(bundle, worktree_path, feedback_text=feedback_text)
                evaluation = evaluator.evaluate(bundle, worktree_path, candidate.patch_text or '', attempt_dir)
                save_attempt_files(attempt_dir, candidate, evaluation)
                case_results.append(evaluation)
                if evaluation.tree_match:
                    break
                feedback_text = evaluation.feedback_message
        finally:
            if not args.keep_worktrees:
                repo_manager.remove_worktree(worktree_path)

        best_similarity = choose_best_result(case_results)
        run_result = CaseRunResult(case=case,
                                   base_commit=bundle.base_commit,
                                   fix_commit=bundle.fix_commit,
                                   attempts=len(case_results),
                                   success=any(item.tree_match for item in case_results),
                                   best_similarity=best_similarity,
                                   artifact_dir=bundle.artifact_dir,
                                   evaluations=case_results)
        prompt_profile.observe(select_learning_evaluation(case_results))
        prompt_profile.save(prompt_profile_path)
        with summary_path.open('a', encoding='utf-8') as fp:
            fp.write(run_result.to_json() + '\n')

    logger.info(f'Saved summary to {summary_path}')


if __name__ == '__main__':
    main()
