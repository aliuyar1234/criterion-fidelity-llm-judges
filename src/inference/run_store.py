from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.common.files import atomic_write_json


class RunStoreError(ValueError):
    """Raised when a resumable M6 run cannot be created or resumed honestly."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _json_loads(payload: str | None) -> Any:
    if payload is None:
        return None
    return json.loads(payload)


class RunStore:
    """SQLite-backed run ledger for one M6 slice."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        metrics_dir: str | Path,
        audit_dir: str | Path,
        spec: Mapping[str, Any],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_dir = Path(metrics_dir)
        self.audit_dir = Path(audit_dir)
        self.spec = dict(spec)
        self.run_id = str(self.spec["run_id"])
        self.sqlite_path = self.run_dir / "run.sqlite"
        self.lock_path = self.run_dir / "lock" / "run.lock"
        self.heartbeat_path = self.run_dir / "heartbeat.json"
        self.control_stop_path = self.run_dir / "control" / "STOP"
        self.resolved_config_path = self.run_dir / "config_resolved.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self._lock_acquired = False
        self.connection = sqlite3.connect(self.sqlite_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
        if self._lock_acquired and self.lock_path.exists():
            self.lock_path.unlink(missing_ok=True)
            self._lock_acquired = False

    def acquire_lock(self, *, force_takeover: bool = False) -> None:
        """Acquire the slice lock or fail fast if another process owns it."""

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "created_at": utc_now_iso(),
        }
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            if not force_takeover:
                raise RunStoreError(
                    "Run lock already exists at "
                    f"{self.lock_path}. Use --force-takeover to replace it."
                ) from error
            self.lock_path.unlink(missing_ok=True)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        self._lock_acquired = True

    def init_or_verify_run(
        self,
        *,
        total_families: int,
        families: list[Mapping[str, Any]],
        resolved_config: Mapping[str, Any],
    ) -> None:
        """Create the run metadata and queue, or verify an existing run matches exactly."""

        row = self.connection.execute(
            "SELECT * FROM run_meta WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        now = utc_now_iso()
        if row is None:
            with self._transaction():
                self.connection.execute(
                    """
                    INSERT INTO run_meta (
                        run_id,
                        milestone,
                        task_family,
                        split,
                        model_id,
                        prompt_id,
                        dataset_path,
                        dataset_sha256,
                        prompt_version,
                        scoring_version,
                        config_fingerprint,
                        git_commit,
                        state,
                        phase,
                        total_families,
                        created_at,
                        updated_at,
                        resolved_config_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.run_id,
                        self.spec["milestone"],
                        self.spec["task_family"],
                        self.spec["split"],
                        self.spec["model_id"],
                        self.spec["prompt_id"],
                        self.spec["dataset_path"],
                        self.spec["dataset_sha256"],
                        self.spec["prompt_version"],
                        self.spec["scoring_version"],
                        self.spec["config_fingerprint"],
                        self.spec["git_commit"],
                        "CREATED",
                        "PREPARE",
                        int(total_families),
                        now,
                        now,
                        _json_dumps(dict(resolved_config)),
                    ),
                )
                self.connection.executemany(
                    """
                    INSERT INTO family_queue (
                        family_id,
                        ordinal,
                        status,
                        attempt_id,
                        started_at,
                        completed_at,
                        duration_ms,
                        retry_count,
                        last_error_code,
                        last_error_msg,
                        stale_recovered
                    ) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, 0, NULL, NULL, 0)
                    """,
                    [
                        (
                            str(family["family_id"]),
                            ordinal,
                            "PENDING",
                        )
                        for ordinal, family in enumerate(families)
                    ],
                )
            atomic_write_json(self.resolved_config_path, dict(resolved_config))
            return

        self._verify_run_compatibility(row, total_families=total_families, families=families)
        atomic_write_json(self.resolved_config_path, dict(resolved_config))

    def start_attempt(self) -> str:
        """Register a new process attempt against this logical run."""

        attempt_id = self.spec["attempt_id_factory"]()
        resume_from_completed = self.completed_count()
        now = utc_now_iso()
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO attempts (
                    attempt_id,
                    run_id,
                    host,
                    pid,
                    started_at,
                    ended_at,
                    status,
                    resume_from_completed,
                    last_heartbeat_at,
                    last_family_id,
                    note
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, NULL)
                """,
                (
                    attempt_id,
                    self.run_id,
                    self.spec["host"],
                    os.getpid(),
                    now,
                    "RUNNING",
                    int(resume_from_completed),
                ),
            )
            self.connection.execute(
                "UPDATE run_meta SET state = ?, phase = ?, updated_at = ? WHERE run_id = ?",
                ("RUNNING", "INFERENCE", now, self.run_id),
            )
        return str(attempt_id)

    def update_attempt_heartbeat(self, attempt_id: str, *, family_id: str | None = None) -> None:
        now = utc_now_iso()
        self.connection.execute(
            """
            UPDATE attempts
            SET last_heartbeat_at = ?, last_family_id = ?
            WHERE attempt_id = ?
            """,
            (now, family_id, attempt_id),
        )
        self.connection.commit()

    def recover_stale_inflight(self) -> int:
        """Return stale inflight families to PENDING so resume can continue cleanly."""

        stale_rows = self.connection.execute(
            """
            SELECT family_id, retry_count, stale_recovered
            FROM family_queue
            WHERE status = 'INFLIGHT'
            """
        ).fetchall()
        if not stale_rows:
            return 0

        now = utc_now_iso()
        with self._transaction():
            for row in stale_rows:
                self.connection.execute(
                    """
                    UPDATE family_queue
                    SET status = 'PENDING',
                        attempt_id = NULL,
                        started_at = NULL,
                        duration_ms = NULL,
                        retry_count = ?,
                        last_error_code = ?,
                        last_error_msg = ?,
                        stale_recovered = ?,
                        completed_at = NULL
                    WHERE family_id = ?
                    """,
                    (
                        int(row["retry_count"]) + 1,
                        "stale_recovered",
                        "Recovered an interrupted inflight family on resume.",
                        int(row["stale_recovered"]) + 1,
                        str(row["family_id"]),
                    ),
                )
            self.connection.execute(
                "UPDATE run_meta SET updated_at = ? WHERE run_id = ?",
                (now, self.run_id),
            )
        return len(stale_rows)

    def pending_family_ids(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT family_id FROM family_queue WHERE status = 'PENDING' ORDER BY ordinal"
        ).fetchall()
        return [str(row["family_id"]) for row in rows]

    def mark_family_inflight(self, family_id: str, attempt_id: str) -> None:
        now = utc_now_iso()
        with self._transaction():
            cursor = self.connection.execute(
                """
                UPDATE family_queue
                SET status = 'INFLIGHT',
                    attempt_id = ?,
                    started_at = ?,
                    completed_at = NULL,
                    duration_ms = NULL
                WHERE family_id = ? AND status = 'PENDING'
                """,
                (attempt_id, now, family_id),
            )
            if cursor.rowcount <= 0:
                raise RunStoreError(
                    f"Cannot mark family {family_id!r} inflight from the current state."
                )
            self.connection.execute(
                "UPDATE run_meta SET state = ?, phase = ?, updated_at = ? WHERE run_id = ?",
                ("RUNNING", "INFERENCE", now, self.run_id),
            )

    def commit_family(
        self,
        *,
        family_id: str,
        attempt_id: str,
        computation: Mapping[str, Any],
        duration_ms: int,
    ) -> None:
        """Atomically persist one family's prompt-, variant-, and family-level outputs."""

        prompt_results = list(computation["prompt_results"])
        variant_results = list(computation["variant_results"])
        family_result = dict(computation["family_result"])
        now = utc_now_iso()

        with self._transaction():
            self.connection.execute("DELETE FROM prompt_results WHERE family_id = ?", (family_id,))
            self.connection.execute("DELETE FROM variant_results WHERE family_id = ?", (family_id,))
            self.connection.execute("DELETE FROM family_results WHERE family_id = ?", (family_id,))

            self.connection.executemany(
                """
                INSERT INTO prompt_results (
                    family_id,
                    variant_id,
                    order_id,
                    displayed_cid_a,
                    displayed_cid_b,
                    label_text_a,
                    label_text_b,
                    label_token_ids_a_json,
                    label_token_ids_b_json,
                    label_token_logprobs_a_json,
                    label_token_logprobs_b_json,
                    logprob_total_a,
                    logprob_total_b,
                    pred_display_label,
                    score_gap,
                    rendered_prefix_text,
                    rendered_prefix_sha256,
                    rendered_prefix_char_len,
                    attempt_id,
                    inference_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["family_id"],
                        row["variant_id"],
                        row["order_id"],
                        row["displayed_cid_a"],
                        row["displayed_cid_b"],
                        row["label_text_a"],
                        row["label_text_b"],
                        _json_dumps(row["label_token_ids_a"]),
                        _json_dumps(row["label_token_ids_b"]),
                        _json_dumps(row["label_token_logprobs_a"]),
                        _json_dumps(row["label_token_logprobs_b"]),
                        float(row["logprob_total_a"]),
                        float(row["logprob_total_b"]),
                        row["pred_display_label"],
                        float(row["score_gap"]),
                        row["rendered_prefix_text"],
                        row["rendered_prefix_sha256"],
                        int(row["rendered_prefix_char_len"]),
                        attempt_id,
                        int(row["inference_ms"]),
                    )
                    for row in prompt_results
                ],
            )

            self.connection.executemany(
                """
                INSERT INTO variant_results (
                    family_id,
                    variant_id,
                    scores_ab_c1,
                    scores_ab_c2,
                    scores_ba_c1,
                    scores_ba_c2,
                    scores_agg_c1,
                    scores_agg_c2,
                    pred_winner_cid,
                    pred_tie,
                    gold_winner_cid,
                    order_pred_ab,
                    order_pred_ba,
                    order_tie_ab,
                    order_tie_ba,
                    order_disagree,
                    attempt_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["family_id"],
                        row["variant_id"],
                        float(row["scores_ab_c1"]),
                        float(row["scores_ab_c2"]),
                        float(row["scores_ba_c1"]),
                        float(row["scores_ba_c2"]),
                        float(row["scores_agg_c1"]),
                        float(row["scores_agg_c2"]),
                        row["pred_winner_cid"],
                        int(row["pred_tie"]),
                        row["gold_winner_cid"],
                        row["order_pred_ab"],
                        row["order_pred_ba"],
                        int(row["order_tie_ab"]),
                        int(row["order_tie_ba"]),
                        int(row["order_disagree"]),
                        attempt_id,
                    )
                    for row in variant_results
                ],
            )

            self.connection.execute(
                """
                INSERT INTO family_results (
                    family_id,
                    task_family,
                    split,
                    base_correct,
                    paraphrase_all_correct,
                    counterfactual_correct,
                    gcf_success,
                    tie_count,
                    variant_count,
                    order_disagreement_count,
                    attempt_id,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    family_id,
                    family_result["task_family"],
                    family_result["split"],
                    int(family_result["base_correct"]),
                    int(family_result["paraphrase_all_correct"]),
                    int(family_result["counterfactual_correct"]),
                    int(family_result["gcf_success"]),
                    int(family_result["tie_count"]),
                    int(family_result["variant_count"]),
                    int(family_result["order_disagreement_count"]),
                    attempt_id,
                    now,
                ),
            )

            self.connection.execute(
                """
                UPDATE family_queue
                SET status = 'COMPLETE',
                    attempt_id = ?,
                    completed_at = ?,
                    duration_ms = ?,
                    last_error_code = NULL,
                    last_error_msg = NULL
                WHERE family_id = ?
                """,
                (attempt_id, now, int(duration_ms), family_id),
            )
            self.connection.execute(
                "UPDATE run_meta SET updated_at = ? WHERE run_id = ?",
                (now, self.run_id),
            )

    def set_state(self, state: str, phase: str) -> None:
        now = utc_now_iso()
        with self._transaction():
            self.connection.execute(
                "UPDATE run_meta SET state = ?, phase = ?, updated_at = ? WHERE run_id = ?",
                (state, phase, now, self.run_id),
            )

    def mark_pause(self, attempt_id: str) -> None:
        now = utc_now_iso()
        with self._transaction():
            self.connection.execute(
                "UPDATE attempts SET status = ?, ended_at = ? WHERE attempt_id = ?",
                ("PAUSED", now, attempt_id),
            )
            self.connection.execute(
                "UPDATE run_meta SET state = ?, updated_at = ? WHERE run_id = ?",
                ("PAUSED", now, self.run_id),
            )

    def mark_completed(self, attempt_id: str) -> None:
        now = utc_now_iso()
        with self._transaction():
            self.connection.execute(
                "UPDATE attempts SET status = ?, ended_at = ? WHERE attempt_id = ?",
                ("COMPLETED", now, attempt_id),
            )
            self.connection.execute(
                "UPDATE run_meta SET state = ?, phase = ?, updated_at = ? WHERE run_id = ?",
                ("COMPLETED", "EXPORT", now, self.run_id),
            )

    def mark_failed(self, attempt_id: str, reason: str) -> None:
        now = utc_now_iso()
        with self._transaction():
            self.connection.execute(
                "UPDATE attempts SET status = ?, ended_at = ?, note = ? WHERE attempt_id = ?",
                ("FAILED", now, reason, attempt_id),
            )
            self.connection.execute(
                "UPDATE run_meta SET state = ?, updated_at = ? WHERE run_id = ?",
                ("FAILED", now, self.run_id),
            )

    def progress_counts(self) -> dict[str, int]:
        row = self.connection.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'COMPLETE' THEN 1 ELSE 0 END) AS completed_families,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS pending_families,
                SUM(CASE WHEN status = 'INFLIGHT' THEN 1 ELSE 0 END) AS inflight_families,
                SUM(
                    CASE WHEN status = 'FAILED_PERMANENT' THEN 1 ELSE 0 END
                ) AS failed_permanent_families
            FROM family_queue
            """
        ).fetchone()
        return {
            key: int(row[key] or 0)
            for key in (
                "completed_families",
                "pending_families",
                "inflight_families",
                "failed_permanent_families",
            )
        }

    def completed_count(self) -> int:
        return self.progress_counts()["completed_families"]

    def pending_count(self) -> int:
        return self.progress_counts()["pending_families"]

    def load_run_meta(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM run_meta WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        if row is None:
            raise RunStoreError(f"Run metadata for {self.run_id!r} is missing.")
        payload = dict(row)
        payload["resolved_config"] = _json_loads(payload.pop("resolved_config_json"))
        return payload

    def fetch_prompt_results(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM prompt_results ORDER BY family_id, variant_id, order_id"
        ).fetchall()
        payload: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            for key in (
                "label_token_ids_a_json",
                "label_token_ids_b_json",
                "label_token_logprobs_a_json",
                "label_token_logprobs_b_json",
            ):
                record[key.removesuffix("_json")] = _json_loads(record.pop(key))
            payload.append(record)
        return payload

    def fetch_variant_results(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM variant_results ORDER BY family_id, variant_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def fetch_family_results(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM family_results ORDER BY family_id").fetchall()
        return [dict(row) for row in rows]

    def status_snapshot(self) -> dict[str, Any]:
        meta = self.load_run_meta()
        counts = self.progress_counts()
        return {
            "run_id": self.run_id,
            "state": meta["state"],
            "phase": meta["phase"],
            "task_family": meta["task_family"],
            "split": meta["split"],
            "model_id": meta["model_id"],
            "prompt_id": meta["prompt_id"],
            "dataset_sha256": meta["dataset_sha256"],
            "total_families": int(meta["total_families"]),
            **counts,
            "heartbeat_path": str(self.heartbeat_path),
            "sqlite_path": str(self.sqlite_path),
        }

    @contextmanager
    def _transaction(self):
        try:
            self.connection.execute("BEGIN")
            yield
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _verify_run_compatibility(
        self,
        row: sqlite3.Row,
        *,
        total_families: int,
        families: Iterable[Mapping[str, Any]],
    ) -> None:
        expected_fields = (
            "milestone",
            "task_family",
            "split",
            "model_id",
            "prompt_id",
            "dataset_path",
            "dataset_sha256",
            "prompt_version",
            "scoring_version",
            "config_fingerprint",
        )
        for field_name in expected_fields:
            if str(row[field_name]) != str(self.spec[field_name]):
                raise RunStoreError(
                    f"Existing run {self.run_id!r} does not match the requested {field_name}."
                )

        if int(row["total_families"]) != int(total_families):
            raise RunStoreError(
                f"Existing run {self.run_id!r} has total_families={row['total_families']} "
                f"but the current dataset has {total_families}."
            )

        queued_ids = [
            str(queued_row["family_id"])
            for queued_row in self.connection.execute(
                "SELECT family_id FROM family_queue ORDER BY ordinal"
            ).fetchall()
        ]
        dataset_ids = [str(family["family_id"]) for family in families]
        if queued_ids != dataset_ids:
            raise RunStoreError(
                f"Existing run {self.run_id!r} does not match the frozen dataset family order."
            )

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                run_id TEXT PRIMARY KEY,
                milestone TEXT NOT NULL,
                task_family TEXT NOT NULL,
                split TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                scoring_version TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                git_commit TEXT,
                state TEXT NOT NULL,
                phase TEXT NOT NULL,
                total_families INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                host TEXT NOT NULL,
                pid INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                resume_from_completed INTEGER NOT NULL,
                last_heartbeat_at TEXT,
                last_family_id TEXT,
                note TEXT,
                FOREIGN KEY (run_id) REFERENCES run_meta(run_id)
            );

            CREATE TABLE IF NOT EXISTS family_queue (
                family_id TEXT PRIMARY KEY,
                ordinal INTEGER UNIQUE NOT NULL,
                status TEXT NOT NULL,
                attempt_id TEXT,
                started_at TEXT,
                completed_at TEXT,
                duration_ms INTEGER,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error_code TEXT,
                last_error_msg TEXT,
                stale_recovered INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_family_queue_status_ordinal
            ON family_queue(status, ordinal);

            CREATE TABLE IF NOT EXISTS prompt_results (
                family_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                displayed_cid_a TEXT NOT NULL,
                displayed_cid_b TEXT NOT NULL,
                label_text_a TEXT NOT NULL,
                label_text_b TEXT NOT NULL,
                label_token_ids_a_json TEXT NOT NULL,
                label_token_ids_b_json TEXT NOT NULL,
                label_token_logprobs_a_json TEXT NOT NULL,
                label_token_logprobs_b_json TEXT NOT NULL,
                logprob_total_a REAL NOT NULL,
                logprob_total_b REAL NOT NULL,
                pred_display_label TEXT NOT NULL,
                score_gap REAL NOT NULL,
                rendered_prefix_text TEXT NOT NULL,
                rendered_prefix_sha256 TEXT NOT NULL,
                rendered_prefix_char_len INTEGER NOT NULL,
                attempt_id TEXT NOT NULL,
                inference_ms INTEGER NOT NULL,
                PRIMARY KEY (family_id, variant_id, order_id)
            );

            CREATE TABLE IF NOT EXISTS variant_results (
                family_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                scores_ab_c1 REAL NOT NULL,
                scores_ab_c2 REAL NOT NULL,
                scores_ba_c1 REAL NOT NULL,
                scores_ba_c2 REAL NOT NULL,
                scores_agg_c1 REAL NOT NULL,
                scores_agg_c2 REAL NOT NULL,
                pred_winner_cid TEXT NOT NULL,
                pred_tie INTEGER NOT NULL,
                gold_winner_cid TEXT NOT NULL,
                order_pred_ab TEXT NOT NULL,
                order_pred_ba TEXT NOT NULL,
                order_tie_ab INTEGER NOT NULL,
                order_tie_ba INTEGER NOT NULL,
                order_disagree INTEGER NOT NULL,
                attempt_id TEXT NOT NULL,
                PRIMARY KEY (family_id, variant_id)
            );

            CREATE TABLE IF NOT EXISTS family_results (
                family_id TEXT PRIMARY KEY,
                task_family TEXT NOT NULL,
                split TEXT NOT NULL,
                base_correct INTEGER NOT NULL,
                paraphrase_all_correct INTEGER NOT NULL,
                counterfactual_correct INTEGER NOT NULL,
                gcf_success INTEGER NOT NULL,
                tie_count INTEGER NOT NULL,
                variant_count INTEGER NOT NULL,
                order_disagreement_count INTEGER NOT NULL,
                attempt_id TEXT NOT NULL,
                completed_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()
