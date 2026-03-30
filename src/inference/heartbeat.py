from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.common.files import atomic_write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class HeartbeatWriter:
    """Cheap operator-facing progress tracker written outside the SQLite ledger."""

    def __init__(
        self,
        *,
        heartbeat_path: str | Path,
        spec: Mapping[str, Any],
        attempt_id: str,
        total_families: int,
        interval_seconds: float = 15.0,
    ) -> None:
        self.heartbeat_path = Path(heartbeat_path)
        self.spec = dict(spec)
        self.attempt_id = attempt_id
        self.interval_seconds = float(interval_seconds)
        self.total_families = int(total_families)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        now = time.time()
        self.state = "CREATED"
        self.phase = "PREPARE"
        self.completed_families = 0
        self.pending_families = int(total_families)
        self.inflight_families = 0
        self.failed_permanent_families = 0
        self.current_family_id: str | None = None
        self.current_variant_id: str | None = None
        self.current_order_id: str | None = None
        self.current_prompt_index: int | None = None
        self.last_step_advance_ts = now
        self.last_family_commit_ts: float | None = None
        self.last_completed_family_id: str | None = None
        self.families_per_hour_ewma = 0.0
        self.prompt_instances_per_sec_ewma = 0.0
        self.stop_requested = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="heartbeat-writer", daemon=True)
        self._thread.start()
        self.write_now()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        self.write_now()

    def set_state(self, state: str, phase: str) -> None:
        with self._lock:
            self.state = state
            self.phase = phase
        self.write_now()

    def update_progress(
        self,
        *,
        family_id: str,
        variant_id: str | None,
        order_id: str | None,
        prompt_index: int | None,
    ) -> None:
        now = time.time()
        with self._lock:
            elapsed = max(now - self.last_step_advance_ts, 1e-6)
            rate = 1.0 / elapsed
            if self.prompt_instances_per_sec_ewma == 0.0:
                self.prompt_instances_per_sec_ewma = rate
            else:
                self.prompt_instances_per_sec_ewma = (0.8 * self.prompt_instances_per_sec_ewma) + (
                    0.2 * rate
                )
            self.current_family_id = family_id
            self.current_variant_id = variant_id
            self.current_order_id = order_id
            self.current_prompt_index = prompt_index
            self.last_step_advance_ts = now

    def update_counts(self, counts: Mapping[str, int]) -> None:
        with self._lock:
            self.completed_families = int(counts["completed_families"])
            self.pending_families = int(counts["pending_families"])
            self.inflight_families = int(counts["inflight_families"])
            self.failed_permanent_families = int(counts["failed_permanent_families"])
        self.write_now()

    def update_commit(self, *, family_id: str, counts: Mapping[str, int]) -> None:
        now = time.time()
        with self._lock:
            elapsed_since_commit = (
                now - self.last_family_commit_ts if self.last_family_commit_ts is not None else None
            )
            if elapsed_since_commit is not None and elapsed_since_commit > 0:
                families_per_hour = 3600.0 / elapsed_since_commit
                if self.families_per_hour_ewma == 0.0:
                    self.families_per_hour_ewma = families_per_hour
                else:
                    self.families_per_hour_ewma = (0.8 * self.families_per_hour_ewma) + (
                        0.2 * families_per_hour
                    )
            self.last_family_commit_ts = now
            self.last_completed_family_id = family_id
            self.current_family_id = family_id
            self.current_variant_id = None
            self.current_order_id = None
            self.current_prompt_index = None
            self.last_step_advance_ts = now
            self.completed_families = int(counts["completed_families"])
            self.pending_families = int(counts["pending_families"])
            self.inflight_families = int(counts["inflight_families"])
            self.failed_permanent_families = int(counts["failed_permanent_families"])
        self.write_now()

    def set_stop_requested(self, requested: bool = True) -> None:
        with self._lock:
            self.stop_requested = requested
        self.write_now()

    def write_now(self) -> None:
        atomic_write_json(self.heartbeat_path, self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            stall_seconds = max(0.0, now - self.last_step_advance_ts)
            expected_prompt_seconds = (
                1.0 / self.prompt_instances_per_sec_ewma
                if self.prompt_instances_per_sec_ewma > 0
                else None
            )
            stalled = bool(
                expected_prompt_seconds is not None
                and stall_seconds > max(120.0, 5.0 * expected_prompt_seconds)
            )
            eta_seconds = (
                int((self.pending_families / (self.families_per_hour_ewma / 3600.0)))
                if self.families_per_hour_ewma > 0 and self.pending_families > 0
                else None
            )
            return {
                "run_id": self.spec["run_id"],
                "attempt_id": self.attempt_id,
                "state": self.state,
                "phase": self.phase,
                "task_family": self.spec["task_family"],
                "split": self.spec["split"],
                "model_id": self.spec["model_id"],
                "prompt_id": self.spec["prompt_id"],
                "dataset_sha256": self.spec["dataset_sha256"],
                "total_families": self.total_families,
                "completed_families": self.completed_families,
                "pending_families": self.pending_families,
                "inflight_families": self.inflight_families,
                "failed_permanent_families": self.failed_permanent_families,
                "current_family_id": self.current_family_id,
                "current_variant_id": self.current_variant_id,
                "current_order_id": self.current_order_id,
                "current_prompt_index": self.current_prompt_index,
                "last_step_advance_at": datetime.fromtimestamp(
                    self.last_step_advance_ts, tz=timezone.utc
                )
                .replace(microsecond=0)
                .isoformat(),
                "last_family_commit_at": (
                    datetime.fromtimestamp(self.last_family_commit_ts, tz=timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    if self.last_family_commit_ts is not None
                    else None
                ),
                "last_completed_family_id": self.last_completed_family_id,
                "families_per_hour_ewma": self.families_per_hour_ewma,
                "prompt_instances_per_sec_ewma": self.prompt_instances_per_sec_ewma,
                "eta_seconds": eta_seconds,
                "stall_seconds": stall_seconds,
                "stalled": stalled,
                "stop_requested": self.stop_requested,
                "updated_at": utc_now_iso(),
            }

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.write_now()
