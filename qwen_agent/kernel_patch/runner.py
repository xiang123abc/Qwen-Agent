from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json5

from .config import KernelPatchConfig, KernelPatchLLMConfig
from .models import KernelPatchRunResult
from .pipeline import KernelPatchPipeline


class KernelPatchDatasetRunner:

    def __init__(self, pipeline: KernelPatchPipeline):
        self.pipeline = pipeline

    def parse_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        for raw_line in Path(dataset_path).read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            cve_id, commit_id = [part.strip() for part in line.split(',', 1)]
            entries.append({
                'cve_id': cve_id,
                'community_commit_id': commit_id,
            })
        return entries

    def run_dataset(self,
                    dataset_path: str,
                    origin_repo: str,
                    target_repo: str,
                    artifacts_root: str,
                    llm_cfg: Optional[KernelPatchLLMConfig] = None,
                    debug_mode: bool = True,
                    max_iterations: int = 2,
                    limit: Optional[int] = None) -> Dict[str, object]:
        results: List[KernelPatchRunResult] = []
        entries = self.parse_dataset(dataset_path)
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            config = KernelPatchConfig(
                cve_id=entry['cve_id'],
                origin_repo=origin_repo,
                target_repo=target_repo,
                community_commit_id=entry['community_commit_id'],
                artifacts_root=artifacts_root,
                debug_mode=debug_mode,
                max_iterations=max_iterations,
                llm=llm_cfg,
            )
            results.append(self.pipeline.run(config))
        summary = self._build_summary(results)
        summary_path = Path(artifacts_root).expanduser().resolve() / 'dataset_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def _build_summary(self, results: Iterable[KernelPatchRunResult]) -> Dict[str, object]:
        results = list(results)
        total = len(results)
        success_count = sum(1 for result in results if result.success)
        failures = [result for result in results if not result.success]
        failure_buckets = Counter()
        consecutive_failures = 0
        max_consecutive_failures = 0
        for result in results:
            if result.success:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
            if result.attempts:
                failure_buckets[result.attempts[-1].apply_check_stderr or 'unknown'] += 1
        return {
            'total': total,
            'success_count': success_count,
            'success_rate': (success_count / total) if total else 0.0,
            'max_consecutive_failures': max_consecutive_failures,
            'failures': [
                {
                    'cve_id': result.cve_id,
                    'patch_path': result.patch_path,
                    'trace_path': result.trace_path,
                    'error': result.attempts[-1].apply_check_stderr if result.attempts else '',
                } for result in failures
            ],
            'failure_patterns': dict(failure_buckets),
            'prompt_strategy_suggestions': self._suggest_prompt_strategies(max_consecutive_failures, failure_buckets),
        }

    def _suggest_prompt_strategies(self, max_consecutive_failures: int,
                                   failure_buckets: Counter) -> List[str]:
        suggestions: List[str] = []
        if max_consecutive_failures >= 2:
            suggestions.append('连续失败较多，建议在 Planner Prompt 中强化“必须解释本地命中依据”和“优先使用旧代码上下文行作为 anchor”。')
        failure_text = '\n'.join(failure_buckets.keys())
        if 'patch does not apply' in failure_text:
            suggestions.append('存在 patch 上下文不匹配，建议在 Solver Prompt 中强调使用 target_snippets 的原始上下文生成 hunk。')
        if 'corrupt patch' in failure_text:
            suggestions.append('存在 patch 格式错误，建议在 Solver Prompt 中强调只输出原始 unified diff，禁止 Markdown 包裹。')
        return suggestions


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Linux kernel CVE patch generation over a dataset file.')
    parser.add_argument('--config', help='JSON/JSON5 config file path')
    parser.add_argument('--dataset', help='Path to dataset file, e.g. cve-use.txt')
    parser.add_argument('--origin-repo')
    parser.add_argument('--target-repo')
    parser.add_argument('--artifacts-root', default='artifacts')
    parser.add_argument('--model',default='qwen3-235b-a22b-thinking-2507')
    parser.add_argument('--api-base')
    parser.add_argument('--api-key')
    parser.add_argument('--model-type', default='oai')
    parser.add_argument('--max-iterations', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no-debug', action='store_true')
    return parser


def _load_config_file(path: str) -> Dict[str, object]:
    return json5.loads(Path(path).read_text(encoding='utf-8'))


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    file_cfg: Dict[str, object] = _load_config_file(args.config) if args.config else {}
    dataset = args.dataset or file_cfg.get('dataset')
    origin_repo = args.origin_repo or file_cfg.get('origin_repo')
    target_repo = args.target_repo or file_cfg.get('target_repo')
    artifacts_root = args.artifacts_root if args.artifacts_root != 'artifacts' else file_cfg.get('artifacts_root',
                                                                                                 'artifacts')
    max_iterations = args.max_iterations if args.max_iterations != 2 else file_cfg.get('max_iterations', 2)
    limit = args.limit if args.limit is not None else file_cfg.get('limit')
    debug_mode = (not args.no_debug) if args.no_debug else file_cfg.get('debug_mode', True)

    if not dataset or not origin_repo or not target_repo:
        parser.error('dataset, origin_repo, and target_repo are required, either via CLI or --config')

    llm_data = dict(file_cfg.get('llm', {}))
    if args.model:
        llm_data['model'] = args.model
    if args.api_base:
        llm_data['api_base'] = args.api_base
    if args.api_key:
        llm_data['api_key'] = args.api_key
    if args.model_type:
        llm_data['model_type'] = args.model_type

    llm_cfg = KernelPatchLLMConfig.model_validate(llm_data) if llm_data else None
    pipeline = KernelPatchPipeline(llm=llm_cfg.to_qwen_cfg() if llm_cfg else None)
    runner = KernelPatchDatasetRunner(pipeline=pipeline)
    summary = runner.run_dataset(
        dataset_path=str(dataset),
        origin_repo=str(origin_repo),
        target_repo=str(target_repo),
        artifacts_root=str(artifacts_root),
        llm_cfg=llm_cfg,
        debug_mode=bool(debug_mode),
        max_iterations=int(max_iterations),
        limit=int(limit) if limit is not None else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
