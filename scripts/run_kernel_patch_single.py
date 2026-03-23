#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import json5

from qwen_agent.kernel_patch.config import KernelPatchConfig
from qwen_agent.kernel_patch.pipeline import KernelPatchPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a single Linux kernel CVE patch task from a JSON/JSON5 config.')
    parser.add_argument('--config', required=True, help='Path to single-task JSON/JSON5 config')
    parser.add_argument('--no-debug', action='store_true')
    args = parser.parse_args()

    config = KernelPatchConfig.model_validate(json5.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.no_debug:
        config.debug_mode = False
    pipeline = KernelPatchPipeline(llm=config.llm.to_qwen_cfg() if config.llm else None)
    result = pipeline.run(config)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
