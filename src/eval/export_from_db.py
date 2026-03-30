from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from src.common.files import atomic_write_json, atomic_write_text
from src.data.constants import TASK_FAMILY_VARIANT_IDS
from src.eval.bootstrap import bootstrap_summary_metrics
from src.eval.metrics import compute_summary_metrics


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    payload = "\n".join(json.dumps(dict(row), ensure_ascii=True) for row in rows)
    if payload:
        payload += "\n"
    atomic_write_text(path, payload)


def _freeze_qc_summary(task_family: str) -> dict[str, Any]:
    qc_path_map = {
        "qa_key": Path("data/processed/qakey/full_freeze_qc.json"),
        "biorubric": Path("data/processed/biorubric/full_freeze_qc.json"),
    }
    return json.loads(qc_path_map[task_family].read_text(encoding="utf-8"))


def _family_results_jsonl(
    family_results: list[Mapping[str, Any]],
    variant_results: list[Mapping[str, Any]],
    prompt_results: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    variants_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompts_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for prompt_row in prompt_results:
        prompts_by_key[
            (str(prompt_row["family_id"]), f"{prompt_row['variant_id']}::{prompt_row['order_id']}")
        ] = dict(prompt_row)

    for variant_row in variant_results:
        family_id = str(variant_row["family_id"])
        prompt_ab = prompts_by_key[(family_id, f"{variant_row['variant_id']}::AB")]
        prompt_ba = prompts_by_key[(family_id, f"{variant_row['variant_id']}::BA")]
        variants_by_family[family_id].append(
            {
                "family_id": family_id,
                "variant_id": variant_row["variant_id"],
                "scores_ab": {"c1": variant_row["scores_ab_c1"], "c2": variant_row["scores_ab_c2"]},
                "scores_ba": {"c1": variant_row["scores_ba_c1"], "c2": variant_row["scores_ba_c2"]},
                "scores_agg": {
                    "c1": variant_row["scores_agg_c1"],
                    "c2": variant_row["scores_agg_c2"],
                },
                "pred_winner_cid": variant_row["pred_winner_cid"],
                "pred_tie": bool(variant_row["pred_tie"]),
                "gold_winner_cid": variant_row["gold_winner_cid"],
                "order_pred_ab": variant_row["order_pred_ab"],
                "order_pred_ba": variant_row["order_pred_ba"],
                "order_tie_ab": bool(variant_row["order_tie_ab"]),
                "order_tie_ba": bool(variant_row["order_tie_ba"]),
                "order_disagree": bool(variant_row["order_disagree"]),
                "label_scores_ab": {
                    "A": prompt_ab["logprob_total_a"],
                    "B": prompt_ab["logprob_total_b"],
                },
                "label_scores_ba": {
                    "A": prompt_ba["logprob_total_a"],
                    "B": prompt_ba["logprob_total_b"],
                },
                "label_token_ids": {
                    "A": prompt_ab["label_token_ids_a"],
                    "B": prompt_ab["label_token_ids_b"],
                },
                "label_token_logprobs": {
                    "A": prompt_ab["label_token_logprobs_a"],
                    "B": prompt_ab["label_token_logprobs_b"],
                },
                "label_token_ids_ba": {
                    "A": prompt_ba["label_token_ids_a"],
                    "B": prompt_ba["label_token_ids_b"],
                },
                "label_token_logprobs_ba": {
                    "A": prompt_ba["label_token_logprobs_a"],
                    "B": prompt_ba["label_token_logprobs_b"],
                },
                "rendered_prefix_ab": prompt_ab["rendered_prefix_text"],
                "rendered_prefix_ba": prompt_ba["rendered_prefix_text"],
                "inference_ms_ab": prompt_ab["inference_ms"],
                "inference_ms_ba": prompt_ba["inference_ms"],
            }
        )

    payload: list[dict[str, Any]] = []
    for family_row in family_results:
        task_family = str(family_row["task_family"])
        variant_order = {
            variant_id: index
            for index, variant_id in enumerate(TASK_FAMILY_VARIANT_IDS[task_family])
        }
        payload.append(
            {
                **dict(family_row),
                "variant_results": sorted(
                    variants_by_family[str(family_row["family_id"])],
                    key=lambda variant_row: variant_order[str(variant_row["variant_id"])],
                ),
            }
        )
    return payload


def write_summary_artifacts(
    *,
    store,
    spec: Mapping[str, Any],
    family_results: list[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Write the summary JSON/CSV artifacts from the current SQLite ledger state."""

    resolved_family_results = (
        list(family_results) if family_results is not None else store.fetch_family_results()
    )
    summary_metrics = compute_summary_metrics(resolved_family_results)
    freeze_qc = _freeze_qc_summary(str(spec["task_family"]))
    run_meta = store.load_run_meta()

    summary_json = store.metrics_dir / "summary_metrics.json"
    summary_csv = store.metrics_dir / "summary_metrics.csv"
    summary_payload = {
        "run_id": spec["run_id"],
        "milestone": spec["milestone"],
        "task_family": spec["task_family"],
        "model_id": spec["model_id"],
        "prompt_id": spec["prompt_id"],
        "split": spec["split"],
        "dataset_path": spec["dataset_path"],
        "dataset_sha256": spec["dataset_sha256"],
        "prompt_version": spec["prompt_version"],
        "scoring_version": spec["scoring_version"],
        "config_fingerprint": spec["config_fingerprint"],
        "git_commit": spec["git_commit"],
        "n_evaluated_families": len(resolved_family_results),
        "summary_metrics": summary_metrics,
        "freeze_qc": {
            "invalid_family_count": freeze_qc["invalid_family_count"],
            "discarded_family_count": freeze_qc["discarded_family_count"],
            "selected_split_counts": freeze_qc["selection_summary"]["selected_split_counts"],
        },
        "run_meta": {
            "state": run_meta["state"],
            "phase": run_meta["phase"],
            "created_at": run_meta["created_at"],
            "updated_at": run_meta["updated_at"],
        },
    }
    atomic_write_json(summary_json, summary_payload)
    _write_csv(summary_csv, [summary_payload["summary_metrics"]], list(summary_metrics.keys()))
    return {
        "summary_metrics_json": str(summary_json),
        "summary_metrics_csv": str(summary_csv),
    }


def export_run_artifacts(
    *,
    store,
    spec: Mapping[str, Any],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, str]:
    """Export CSV/JSON/JSONL artifacts deterministically from one SQLite slice ledger."""

    prompt_results = store.fetch_prompt_results()
    variant_results = store.fetch_variant_results()
    family_results = store.fetch_family_results()
    bootstrap_ci = bootstrap_summary_metrics(
        family_results,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    raw_output_jsonl = store.run_dir / "family_results.jsonl"
    prompt_csv = store.metrics_dir / "prompt_results.csv"
    variant_csv = store.metrics_dir / "variant_results.csv"
    family_csv = store.metrics_dir / "family_results.csv"
    bootstrap_json = store.metrics_dir / "bootstrap_ci.json"

    _write_jsonl(
        raw_output_jsonl,
        _family_results_jsonl(
            family_results=family_results,
            variant_results=variant_results,
            prompt_results=prompt_results,
        ),
    )

    _write_csv(
        prompt_csv,
        prompt_results,
        fieldnames=list(prompt_results[0].keys()) if prompt_results else [],
    )
    _write_csv(
        variant_csv,
        variant_results,
        fieldnames=list(variant_results[0].keys()) if variant_results else [],
    )
    _write_csv(
        family_csv,
        family_results,
        fieldnames=list(family_results[0].keys()) if family_results else [],
    )

    atomic_write_json(bootstrap_json, bootstrap_ci)
    summary_paths = write_summary_artifacts(
        store=store,
        spec=spec,
        family_results=family_results,
    )

    return {
        "raw_output_jsonl": str(raw_output_jsonl),
        "prompt_results_csv": str(prompt_csv),
        "variant_results_csv": str(variant_csv),
        "family_results_csv": str(family_csv),
        **summary_paths,
        "bootstrap_ci_json": str(bootstrap_json),
    }
