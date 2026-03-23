from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from qwen_agent.kernel_patch.models import StepTrace, ToolTraceEntry


class TraceRecorder:

    def __init__(self, debug_log_path: Path):
        self.debug_log_path = debug_log_path
        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[Dict[str, Any]] = []
        self._steps: List[StepTrace] = []

    def record_event(self, phase: str, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'phase': phase,
            'event_type': event_type,
            'payload': payload,
        }
        self._events.append(event)
        with self.debug_log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    def add_step(self,
                 step: str,
                 prompt: Optional[str],
                 output: Any,
                 tool_calls: Optional[List[ToolTraceEntry]] = None) -> None:
        self._steps.append(
            StepTrace(
                step=step,
                prompt=prompt,
                output=output,
                tool_calls=tool_calls or [],
            ))

    def build_trace_payload(self, task: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            'task': task,
            'steps': [step.model_dump() for step in self._steps],
            'events': list(self._events),
        }
        if extra:
            payload.update(extra)
        return payload
