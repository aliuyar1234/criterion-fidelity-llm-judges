from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

from src.common.files import sha256_file, sha256_json
from src.data.toy_records import TOY_QA_FAMILY
from src.eval.main_results_table import build_standard_only_artifacts
from src.inference.full_runner import run_standard_slice
from src.inference.heartbeat import HeartbeatWriter
from src.inference.run_store import RunStore
from src.inference.score_ab import MockContinuationScorer, render_variant_prefix


class DummyTokenizer:
    is_fast = False

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        if add_generation_prompt:
            rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)


def _build_consistently_correct_scorer(families):
    tokenizer = DummyTokenizer()
    scores_by_prefix: dict[str, dict[str, float]] = {}
    for family in families:
        for variant in family["variants"]:
            for order, label_scores in (
                ("AB", {"A": -0.1, "B": -1.0}),
                ("BA", {"A": -1.0, "B": -0.1}),
            ):
                prefix = render_variant_prefix(
                    family=family,
                    variant=variant,
                    tokenizer=tokenizer,
                    prompt_name="standard",
                    order=order,
                )
                if variant["variant_id"] == "counterfactual":
                    label_scores = (
                        {"A": -1.0, "B": -0.1} if order == "AB" else {"A": -0.1, "B": -1.0}
                    )
                scores_by_prefix[prefix] = label_scores
    return tokenizer, MockContinuationScorer(scores_by_prefix=scores_by_prefix)


def _write_dataset(path: Path, families: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(family, ensure_ascii=True) for family in families) + "\n",
        encoding="utf-8",
    )
    return path


def _build_slice_spec(tmp_path: Path, dataset_path: Path, *, run_id: str) -> dict[str, object]:
    attempt_counter = {"value": 0}

    def _attempt_id_factory() -> str:
        attempt_counter["value"] += 1
        return f"attempt-{attempt_counter['value']}"

    resolved_config = {
        "experiment_config_path": "tests/full_standard.json",
        "experiment_config": {"stage": "full_inference"},
        "model_config_path": "tests/model.json",
        "model_config": {"model_id": "Llama-3.1-8B-Instruct"},
        "prompt_config_path": "tests/prompt.json",
        "prompt_config": {"prompt_id": "standard"},
        "task_family": "qa_key",
        "split": "test",
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "scoring_version": "continuation_ab_v1",
        "seed": 2026,
    }
    return {
        "run_id": run_id,
        "milestone": "M6",
        "task_family": "qa_key",
        "split": "test",
        "model_id": "Llama-3.1-8B-Instruct",
        "prompt_id": "standard",
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "prompt_version": "v1",
        "scoring_version": "continuation_ab_v1",
        "config_fingerprint": sha256_json(resolved_config),
        "git_commit": "not_a_git_repo",
        "run_dir": str(tmp_path / "results" / "raw_outputs" / "m6_standard" / run_id),
        "metrics_dir": str(tmp_path / "results" / "metrics" / "m6_standard" / run_id),
        "audit_dir": str(tmp_path / "data" / "audits" / "m6_standard" / run_id),
        "model_config_path": "tests/model.json",
        "prompt_config_path": "tests/prompt.json",
        "exp_config_path": "tests/full_standard.json",
        "global_seed": 2026,
        "bootstrap_replicates": 50,
        "heartbeat_interval_seconds": 0.05,
        "host": "test-host",
        "attempt_id_factory": _attempt_id_factory,
        "resolved_config": resolved_config,
    }


def test_run_store_recovers_stale_inflight_family() -> None:
    tmp_path = (Path(".pytest_local_tmp") / "m6_run_store").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    family = deepcopy(TOY_QA_FAMILY)
    family["split"] = "test"
    dataset_path = _write_dataset(
        tmp_path / "data" / "processed" / "qakey" / "test_families.jsonl", [family]
    )
    spec = _build_slice_spec(tmp_path, dataset_path, run_id="m6__stale_recovery")

    store = RunStore(
        run_dir=spec["run_dir"],
        metrics_dir=spec["metrics_dir"],
        audit_dir=spec["audit_dir"],
        spec=spec,
    )
    try:
        store.acquire_lock()
        store.init_or_verify_run(
            total_families=1,
            families=[family],
            resolved_config=spec["resolved_config"],
        )
        attempt_id = store.start_attempt()
        store.mark_family_inflight(str(family["family_id"]), attempt_id)
    finally:
        store.close()

    store = RunStore(
        run_dir=spec["run_dir"],
        metrics_dir=spec["metrics_dir"],
        audit_dir=spec["audit_dir"],
        spec=spec,
    )
    try:
        store.acquire_lock()
        store.init_or_verify_run(
            total_families=1,
            families=[family],
            resolved_config=spec["resolved_config"],
        )
        recovered = store.recover_stale_inflight()
        assert recovered == 1
        assert store.pending_family_ids() == [family["family_id"]]
    finally:
        store.close()


def test_run_standard_slice_supports_pause_resume_and_exports() -> None:
    tmp_path = (Path(".pytest_local_tmp") / "m6_pause_resume").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    family_1 = deepcopy(TOY_QA_FAMILY)
    family_1["split"] = "test"
    family_2 = deepcopy(TOY_QA_FAMILY)
    family_2["family_id"] = "qakey_demo_0002"
    family_2["split"] = "test"

    dataset_path = _write_dataset(
        tmp_path / "data" / "processed" / "qakey" / "test_families.jsonl",
        [family_1, family_2],
    )
    spec = _build_slice_spec(tmp_path, dataset_path, run_id="m6__pause_resume")
    tokenizer, scorer = _build_consistently_correct_scorer([family_1, family_2])

    paused = run_standard_slice(
        spec=spec,
        tokenizer=tokenizer,
        scorer=scorer,
        stop_after_families=1,
    )
    assert paused["state"] == "PAUSED"
    assert paused["completed_families"] == 1
    assert paused["pending_families"] == 1

    completed = run_standard_slice(
        spec=spec,
        tokenizer=tokenizer,
        scorer=scorer,
    )
    assert completed["state"] == "COMPLETED"
    assert completed["completed_families"] == 2
    assert completed["pending_families"] == 0

    summary_path = Path(completed["summary_metrics_json"])
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["n_evaluated_families"] == 2
    assert summary_payload["summary_metrics"]["gcf"] == 1.0
    assert summary_payload["run_meta"]["state"] == "COMPLETED"
    assert summary_payload["run_meta"]["phase"] == "EXPORT"

    figure_paths = build_standard_only_artifacts(
        metrics_root=tmp_path / "results" / "metrics" / "m6_standard",
        output_dir=tmp_path / "results" / "figures" / "m6_standard",
    )
    assert Path(figure_paths["main_results_csv"]).exists()
    assert Path(figure_paths["baseacc_vs_gcf_figure"]).exists()


def test_heartbeat_writer_exposes_progress_and_stop_state() -> None:
    tmp_path = (Path(".pytest_local_tmp") / "m6_heartbeat").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    heartbeat = HeartbeatWriter(
        heartbeat_path=tmp_path / "heartbeat.json",
        spec={
            "run_id": "m6__heartbeat_demo",
            "task_family": "qa_key",
            "split": "test",
            "model_id": "Llama-3.1-8B-Instruct",
            "prompt_id": "standard",
            "dataset_sha256": "deadbeef",
        },
        attempt_id="attempt-1",
        total_families=2,
        interval_seconds=60.0,
    )
    heartbeat.set_state("RUNNING", "INFERENCE")
    heartbeat.update_progress(
        family_id="qakey_demo_0001",
        variant_id="base",
        order_id="AB",
        prompt_index=0,
    )
    heartbeat.update_commit(
        family_id="qakey_demo_0001",
        counts={
            "completed_families": 1,
            "pending_families": 1,
            "inflight_families": 0,
            "failed_permanent_families": 0,
        },
    )
    heartbeat.set_stop_requested(True)
    heartbeat.write_now()

    payload = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert payload["state"] == "RUNNING"
    assert payload["phase"] == "INFERENCE"
    assert payload["completed_families"] == 1
    assert payload["pending_families"] == 1
    assert payload["last_completed_family_id"] == "qakey_demo_0001"
    assert payload["stop_requested"] is True
