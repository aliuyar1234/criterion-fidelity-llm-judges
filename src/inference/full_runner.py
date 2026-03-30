from __future__ import annotations

import gc
import json
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common.artifacts import slugify_component
from src.common.config import load_config
from src.common.files import atomic_write_json, sha256_file, sha256_json
from src.data.canonical_tables import load_jsonl_records
from src.data.constants import FAMILY_SPLITS, PRIMARY_TASK_FAMILIES
from src.data.schema_validation import validate_family_record
from src.eval.audit_sample_from_db import build_post_run_failure_audit
from src.eval.export_from_db import export_run_artifacts, write_summary_artifacts
from src.inference.control import RunControl
from src.inference.family_runner import score_family
from src.inference.heartbeat import HeartbeatWriter
from src.inference.run_store import RunStore
from src.inference.score_ab import PRIMARY_MODEL_REPO_IDS, TransformersContinuationScorer

DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_GLOBAL_SEED = 2026
SCORING_VERSION = "continuation_ab_v1"


class FullBaselineRunError(ValueError):
    """Raised when a full baseline slice cannot be run honestly."""


FullStandardRunError = FullBaselineRunError


def _git_commit_or_none() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "not_a_git_repo"
    return completed.stdout.strip()


def _attempt_id_factory() -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rand_suffix = f"{random.randrange(16**6):06x}"
    return f"{timestamp}__{socket.gethostname()}__pid{os.getpid()}__{rand_suffix}"


def _dataset_path_for(task_family: str, split: str) -> Path:
    task_dir = {"qa_key": "qakey", "biorubric": "biorubric"}[task_family]
    return Path("data/processed") / task_dir / f"{split}_families.jsonl"


def load_family_records(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    """Load and validate frozen family records for one slice."""

    families = load_jsonl_records(path)
    validated: list[dict[str, Any]] = []
    for family in families:
        validate_family_record(family)
        if split is None or family["split"] == split:
            validated.append(dict(family))
    if not validated:
        raise FullBaselineRunError(
            "No validated family records were available for the requested slice."
        )
    return validated


def resolve_slice_specs(
    *,
    exp_config_path: str | Path,
    family_type: str | None = None,
    model_id: str | None = None,
    prompt_id: str | None = None,
    split_override: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve the locked full-baseline slices from the experiment config and optional filters."""

    exp_config = load_config(exp_config_path)
    milestone = str(exp_config.get("milestone", "M6"))
    run_group = str(exp_config.get("run_group", f"{milestone.lower()}_baseline"))
    run_prefix = str(exp_config.get("run_prefix", milestone.lower()))
    split = split_override or exp_config.get("split")
    if split not in FAMILY_SPLITS:
        raise FullBaselineRunError("Full-baseline config split must be one of train/dev/test.")

    prompt_config_path = exp_config.get("prompt_config")
    if not isinstance(prompt_config_path, str) or not prompt_config_path:
        raise FullBaselineRunError("Full-baseline config must contain prompt_config.")
    prompt_config = load_config(prompt_config_path)
    resolved_prompt_id = prompt_config.get("prompt_id")
    if not isinstance(resolved_prompt_id, str) or not resolved_prompt_id:
        raise FullBaselineRunError("Prompt config must contain a non-empty prompt_id.")

    if prompt_id is not None and prompt_id != resolved_prompt_id:
        return []

    model_config_paths = exp_config.get("models")
    if not isinstance(model_config_paths, list) or not model_config_paths:
        raise FullBaselineRunError("Full-baseline config must contain a non-empty models list.")

    task_families = exp_config.get("task_families")
    if not isinstance(task_families, list) or not task_families:
        raise FullBaselineRunError(
            "Full-baseline config must contain a non-empty task_families list."
        )

    filtered_task_families = [
        str(task_family_name)
        for task_family_name in task_families
        if str(task_family_name) in PRIMARY_TASK_FAMILIES
        and (family_type is None or family_type == str(task_family_name))
    ]
    if not filtered_task_families:
        return []

    seed = int(exp_config.get("global_seed", DEFAULT_GLOBAL_SEED))
    bootstrap_replicates = int(exp_config.get("bootstrap_replicates", DEFAULT_BOOTSTRAP_REPLICATES))
    heartbeat_interval_seconds = float(
        exp_config.get("heartbeat_interval_seconds", DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
    )
    prompt_version = str(prompt_config.get("schema_version", "v1"))
    git_commit = _git_commit_or_none()

    slice_specs: list[dict[str, Any]] = []
    for model_config_path in model_config_paths:
        model_config = load_config(model_config_path)
        resolved_model_id = model_config.get("model_id")
        if not isinstance(resolved_model_id, str) or not resolved_model_id:
            raise FullBaselineRunError("Model config must contain a non-empty model_id.")
        if resolved_model_id not in PRIMARY_MODEL_REPO_IDS:
            raise FullBaselineRunError(
                f"Unsupported model_id {resolved_model_id!r} for {milestone}."
            )
        if model_id is not None and model_id != resolved_model_id:
            continue

        for task_family_name in filtered_task_families:
            dataset_path = _dataset_path_for(task_family_name, split)
            dataset_sha256 = sha256_file(dataset_path)
            config_fingerprint = sha256_json(
                {
                    "experiment": exp_config,
                    "model": model_config,
                    "prompt": prompt_config,
                    "task_family": task_family_name,
                    "split": split,
                }
            )
            run_id = "__".join(
                [
                    slugify_component(run_prefix),
                    slugify_component(task_family_name),
                    slugify_component(split),
                    slugify_component(resolved_prompt_id),
                    slugify_component(resolved_model_id),
                    f"ds{dataset_sha256[:8]}",
                    f"pv{slugify_component(prompt_version)}",
                    f"sv{slugify_component(SCORING_VERSION)}",
                ]
            )
            run_root = Path("results/raw_outputs") / run_group / run_id
            slice_specs.append(
                {
                    "run_id": run_id,
                    "milestone": milestone,
                    "run_group": run_group,
                    "run_prefix": run_prefix,
                    "task_family": task_family_name,
                    "split": split,
                    "model_id": resolved_model_id,
                    "prompt_id": resolved_prompt_id,
                    "dataset_path": str(dataset_path),
                    "dataset_sha256": dataset_sha256,
                    "prompt_version": prompt_version,
                    "scoring_version": SCORING_VERSION,
                    "config_fingerprint": config_fingerprint,
                    "git_commit": git_commit,
                    "run_dir": str(run_root),
                    "metrics_dir": str(Path("results/metrics") / run_group / run_id),
                    "audit_dir": str(Path("data/audits") / run_group / run_id),
                    "model_config_path": str(model_config_path),
                    "prompt_config_path": str(prompt_config_path),
                    "exp_config_path": str(exp_config_path),
                    "global_seed": seed,
                    "bootstrap_replicates": bootstrap_replicates,
                    "heartbeat_interval_seconds": heartbeat_interval_seconds,
                    "host": socket.gethostname(),
                    "attempt_id_factory": _attempt_id_factory,
                    "resolved_config": {
                        "experiment_config_path": str(exp_config_path),
                        "experiment_config": exp_config,
                        "model_config_path": str(model_config_path),
                        "model_config": model_config,
                        "prompt_config_path": str(prompt_config_path),
                        "prompt_config": prompt_config,
                        "task_family": task_family_name,
                        "split": split,
                        "dataset_path": str(dataset_path),
                        "dataset_sha256": dataset_sha256,
                        "scoring_version": SCORING_VERSION,
                        "seed": seed,
                    },
                }
            )

    return slice_specs


def load_model_bundle(
    model_config_path: str | Path, *, local_files_only: bool = False
) -> dict[str, Any]:
    """Load one primary model/tokenizer bundle for sequence continuation scoring."""

    model_config = load_config(model_config_path)
    model_id = str(model_config["model_id"])
    repo_id = PRIMARY_MODEL_REPO_IDS[model_id]
    load_in_4bit = bool(model_config.get("load_in_4bit", False))

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        local_files_only=local_files_only,
        use_fast=True,
    )
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        local_files_only=local_files_only,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    scorer = TransformersContinuationScorer(model=model, tokenizer=tokenizer)
    return {
        "model_id": model_id,
        "tokenizer": tokenizer,
        "scorer": scorer,
        "model": model,
    }


def release_model_bundle(model_bundle: Mapping[str, Any]) -> None:
    model = model_bundle.get("model")
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_matrix_status(slice_specs: list[Mapping[str, Any]]) -> Path:
    """Write a lightweight matrix-status view across all configured baseline slices."""

    payload = {"slices": {}}
    for spec in slice_specs:
        run_dir = Path(spec["run_dir"])
        heartbeat_path = run_dir / "heartbeat.json"
        if heartbeat_path.exists():
            payload["slices"][spec["run_id"]] = json.loads(
                heartbeat_path.read_text(encoding="utf-8")
            )
        else:
            payload["slices"][spec["run_id"]] = {
                "run_id": spec["run_id"],
                "state": "NOT_STARTED",
                "phase": "PREPARE",
                "task_family": spec["task_family"],
                "split": spec["split"],
                "model_id": spec["model_id"],
                "prompt_id": spec["prompt_id"],
                "dataset_sha256": spec["dataset_sha256"],
            }
    run_group = str(slice_specs[0]["run_group"])
    destination = Path("results/raw_outputs") / run_group / "matrix_status.json"
    atomic_write_json(destination, payload)
    return destination


def run_standard_slice(
    *,
    spec: Mapping[str, Any],
    tokenizer: Any | None,
    scorer: Any | None,
    only_postprocess: bool = False,
    stop_after_families: int | None = None,
    force_takeover: bool = False,
) -> dict[str, Any]:
    """Run or resume one full-baseline slice."""

    families = load_family_records(spec["dataset_path"], split=spec["split"])
    store = RunStore(
        run_dir=spec["run_dir"],
        metrics_dir=spec["metrics_dir"],
        audit_dir=spec["audit_dir"],
        spec=spec,
    )
    control = RunControl(stop_file_path=store.control_stop_path)
    control.install_signal_handlers()

    try:
        store.acquire_lock(force_takeover=force_takeover)
        store.init_or_verify_run(
            total_families=len(families),
            families=families,
            resolved_config=spec["resolved_config"],
        )
        recovered_stale_families = store.recover_stale_inflight()
        attempt_id = store.start_attempt()
        counts = store.progress_counts()
        heartbeat = HeartbeatWriter(
            heartbeat_path=store.heartbeat_path,
            spec=spec,
            attempt_id=attempt_id,
            total_families=len(families),
            interval_seconds=float(spec["heartbeat_interval_seconds"]),
        )
        heartbeat.set_state("RUNNING", "INFERENCE")
        heartbeat.update_counts(counts)
        heartbeat.start()

        try:
            if not only_postprocess:
                if tokenizer is None or scorer is None:
                    raise FullBaselineRunError(
                        "Tokenizer and scorer are required unless --only-postprocess is used."
                    )

                processed_in_attempt = 0
                pending_ids = set(store.pending_family_ids())
                for family in families:
                    family_id = str(family["family_id"])
                    if family_id not in pending_ids:
                        continue

                    control.poll_stop_file()
                    if (
                        stop_after_families is not None
                        and processed_in_attempt >= stop_after_families
                    ):
                        control.request_stop()
                    if control.hard_abort_requested:
                        raise KeyboardInterrupt("Immediate abort requested.")
                    if control.stop_requested:
                        heartbeat.set_stop_requested(True)
                        store.mark_pause(attempt_id)
                        heartbeat.set_state("PAUSED", "INFERENCE")
                        control.consume_stop_file()
                        return {
                            "run_id": spec["run_id"],
                            "state": "PAUSED",
                            "recovered_stale_families": recovered_stale_families,
                            "completed_families": store.completed_count(),
                            "pending_families": store.pending_count(),
                        }

                    store.mark_family_inflight(family_id, attempt_id)
                    start_time = time.perf_counter()
                    computation = score_family(
                        family=family,
                        tokenizer=tokenizer,
                        scorer=scorer,
                        prompt_name=str(spec["prompt_id"]),
                        progress_callback=lambda fid, vid, oid, prompt_index: (
                            heartbeat.update_progress(
                                family_id=fid,
                                variant_id=vid,
                                order_id=oid,
                                prompt_index=prompt_index,
                            ),
                            store.update_attempt_heartbeat(attempt_id, family_id=fid),
                        ),
                    )
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    store.commit_family(
                        family_id=family_id,
                        attempt_id=attempt_id,
                        computation=computation,
                        duration_ms=duration_ms,
                    )
                    counts = store.progress_counts()
                    heartbeat.update_commit(family_id=family_id, counts=counts)
                    store.update_attempt_heartbeat(attempt_id, family_id=family_id)
                    processed_in_attempt += 1

                    control.poll_stop_file()
                    if control.hard_abort_requested:
                        raise KeyboardInterrupt("Immediate abort requested.")
                    if (
                        stop_after_families is not None
                        and processed_in_attempt >= stop_after_families
                    ):
                        control.request_stop()
                    if control.stop_requested:
                        heartbeat.set_stop_requested(True)
                        store.mark_pause(attempt_id)
                        heartbeat.set_state("PAUSED", "INFERENCE")
                        control.consume_stop_file()
                        return {
                            "run_id": spec["run_id"],
                            "state": "PAUSED",
                            "recovered_stale_families": recovered_stale_families,
                            "completed_families": store.completed_count(),
                            "pending_families": store.pending_count(),
                        }

            heartbeat.set_state("POSTPROCESS", "BOOTSTRAP")
            store.set_state("POSTPROCESS", "BOOTSTRAP")
            export_paths = export_run_artifacts(
                store=store,
                spec=spec,
                n_bootstrap=int(spec["bootstrap_replicates"]),
                bootstrap_seed=int(spec["global_seed"]),
            )
            heartbeat.set_state("POSTPROCESS", "AUDIT_SAMPLE")
            store.set_state("POSTPROCESS", "AUDIT_SAMPLE")
            audit_paths = build_post_run_failure_audit(
                store=store,
                spec=spec,
                sample_seed=int(spec["global_seed"]),
            )
            heartbeat.set_state("POSTPROCESS", "EXPORT")
            store.set_state("POSTPROCESS", "EXPORT")
            store.mark_completed(attempt_id)
            heartbeat.set_state("COMPLETED", "EXPORT")
            export_paths.update(
                write_summary_artifacts(
                    store=store,
                    spec=spec,
                )
            )
            return {
                "run_id": spec["run_id"],
                "state": "COMPLETED",
                "recovered_stale_families": recovered_stale_families,
                "completed_families": store.completed_count(),
                "pending_families": store.pending_count(),
                **export_paths,
                **audit_paths,
            }
        except KeyboardInterrupt as error:
            store.mark_failed(attempt_id, str(error))
            heartbeat.set_state("FAILED", "INFERENCE")
            raise
        except Exception as error:
            store.mark_failed(attempt_id, repr(error))
            heartbeat.set_state("FAILED", "INFERENCE")
            raise
        finally:
            heartbeat.stop()
    finally:
        store.close()


def group_slice_specs_by_model(
    slice_specs: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in slice_specs:
        grouped[str(spec["model_config_path"])].append(dict(spec))
    return dict(grouped)
