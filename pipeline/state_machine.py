from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable


class StageStatus(str, Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"


@dataclass
class StageResult:
    stage: str
    status: StageStatus
    attempts: int
    started_at: str
    finished_at: str
    output: dict[str, object] | None = None
    error: str | None = None


class PipelineRunner:
    def __init__(
        self,
        *,
        run_id: str,
        state_path: Path,
        latest_state_path: Path | None,
        max_retries: int,
        retry_delay_seconds: float,
    ) -> None:
        self.run_id = run_id
        self.state_path = state_path
        self.latest_state_path = latest_state_path
        self.max_retries = max(0, max_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._state: dict[str, object] = {
            "run_id": run_id,
            "status": "running",
            "current_stage": None,
            "started_at": self._now(),
            "finished_at": None,
            "stages": {},
        }
        self._persist_state()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        if self.latest_state_path is not None:
            self.latest_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.latest_state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _set_stage_status(
        self,
        *,
        stage: str,
        status: StageStatus,
        attempts: int,
        started_at: str,
        finished_at: str | None = None,
        output: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        stages = self._state.setdefault("stages", {})
        assert isinstance(stages, dict)
        stages[stage] = {
            "status": status.value,
            "attempts": attempts,
            "started_at": started_at,
            "finished_at": finished_at,
            "output": output,
            "error": error,
        }
        self._state["current_stage"] = stage if status == StageStatus.running else None
        self._persist_state()

    def run_stage(self, stage: str, action: Callable[[], dict[str, object]]) -> StageResult:
        attempts = 0
        started_at = self._now()

        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            self._set_stage_status(
                stage=stage,
                status=StageStatus.running,
                attempts=attempt,
                started_at=started_at,
            )
            try:
                output = action() or {}
                finished_at = self._now()
                self._set_stage_status(
                    stage=stage,
                    status=StageStatus.success,
                    attempts=attempt,
                    started_at=started_at,
                    finished_at=finished_at,
                    output=output,
                )
                return StageResult(
                    stage=stage,
                    status=StageStatus.success,
                    attempts=attempt,
                    started_at=started_at,
                    finished_at=finished_at,
                    output=output,
                )
            except Exception as exc:
                finished_at = self._now()
                error = str(exc)
                self._set_stage_status(
                    stage=stage,
                    status=StageStatus.failed,
                    attempts=attempt,
                    started_at=started_at,
                    finished_at=finished_at,
                    error=error,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue
                return StageResult(
                    stage=stage,
                    status=StageStatus.failed,
                    attempts=attempts,
                    started_at=started_at,
                    finished_at=finished_at,
                    error=error,
                )

        raise RuntimeError(f"Stage {stage} reached an invalid state")

    def finalize(self, *, success: bool, error: str | None = None) -> None:
        self._state["status"] = "success" if success else "failed"
        self._state["error"] = error
        self._state["finished_at"] = self._now()
        self._state["current_stage"] = None
        self._persist_state()
