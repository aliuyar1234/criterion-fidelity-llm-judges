from __future__ import annotations

import csv
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.common.files import atomic_write_json
from src.data.canonical_tables import load_jsonl_records
from src.data.constants import TASK_FAMILY_VARIANT_IDS

FAILURE_SAMPLE_SEED = 2026
POST_RUN_FAILURE_TARGET = 50


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_family_map(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(family["family_id"]): dict(family) for family in load_jsonl_records(dataset_path)}


def _primary_failure_variant(
    task_family: str, variant_rows: Mapping[str, Mapping[str, Any]]
) -> str:
    for variant_id in TASK_FAMILY_VARIANT_IDS[task_family]:
        variant_row = variant_rows[variant_id]
        if str(variant_row["pred_winner_cid"]) != str(variant_row["gold_winner_cid"]):
            return variant_id
    raise ValueError("A GCF-failing family did not expose a failing logical variant.")


def _derive_primary_label(
    *,
    task_family: str,
    family: Mapping[str, Any],
    primary_variant: Mapping[str, Any],
) -> tuple[str, str]:
    if task_family == "qa_key":
        donor_checks = family["metadata"]["selected_donor_checks"]
        has_data_bug = not (
            bool(family["metadata"]["qc_passed"]) and bool(donor_checks["is_admissible"])
        )
        if has_data_bug:
            return ("data_bug", "Locked QA-Key QC or donor admissibility checks failed.")
    elif task_family == "biorubric":
        checks = family["metadata"]["candidate_checks"]
        has_data_bug = not all(
            [
                checks["both_candidates_factual_by_source"],
                checks["shared_sentence_identical"],
                checks["same_distinguishing_template"],
                checks["c1_exactly_two_sentences"],
                checks["c2_exactly_two_sentences"],
                checks["no_extra_facts"],
                checks["candidate_length_ratio_in_range"],
            ]
        )
        if has_data_bug:
            return ("data_bug", "At least one locked BioRubric candidate-validity check failed.")
    else:  # pragma: no cover - guarded by the frozen task-family set
        raise ValueError(f"Unsupported task family for audit derivation: {task_family}")

    if int(primary_variant["pred_tie"]):
        return ("model_tie", "Mirrored-order aggregate tied within the locked epsilon.")
    if int(primary_variant["order_disagree"]):
        return (
            "order_effect",
            "Failure driven by AB/BA disagreement on the primary failing variant.",
        )
    return (
        "genuine_criterion_failure",
        "No higher-priority locked audit label applied on review of the failing family.",
    )


def build_post_run_failure_audit(
    *,
    store,
    spec: Mapping[str, Any],
    sample_seed: int = FAILURE_SAMPLE_SEED,
    target_failures: int = POST_RUN_FAILURE_TARGET,
) -> dict[str, str]:
    """Write the locked post-run failure audit CSV and summary for one completed slice."""

    family_results = [row for row in store.fetch_family_results() if int(row["gcf_success"]) == 0]
    n_failed_families_available = len(family_results)
    variant_results_all = store.fetch_variant_results()
    variant_rows_by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for variant_row in variant_results_all:
        variant_rows_by_family.setdefault(str(variant_row["family_id"]), {})[
            str(variant_row["variant_id"])
        ] = dict(variant_row)

    if len(family_results) > target_failures:
        rng = random.Random(sample_seed)
        family_results = rng.sample(family_results, target_failures)
        family_results.sort(key=lambda row: str(row["family_id"]))

    family_map = _load_family_map(spec["dataset_path"])
    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    paraphrase_ids = TASK_FAMILY_VARIANT_IDS[str(spec["task_family"])][1:3]
    fieldnames = [
        "audit_id",
        "run_id",
        "task_family",
        "model_id",
        "prompt_id",
        "split",
        "family_id",
        "sample_seed",
        "sample_reason",
        "base_correct",
        "para_1_or_lex_correct",
        "para_2_or_struct_correct",
        "counterfactual_correct",
        "base_tie",
        "para_1_or_lex_tie",
        "para_2_or_struct_tie",
        "counterfactual_tie",
        "base_order_disagree",
        "para_1_or_lex_order_disagree",
        "para_2_or_struct_order_disagree",
        "counterfactual_order_disagree",
        "primary_failure_variant",
        "pred_base_cid",
        "pred_para_1_or_lex_cid",
        "pred_para_2_or_struct_cid",
        "pred_counterfactual_cid",
        "gold_base_cid",
        "gold_counterfactual_cid",
        "audit_complete",
        "primary_label",
        "notes",
        "annotator_id",
        "timestamp_utc",
    ]

    audit_rows: list[dict[str, Any]] = []
    for family_row in family_results:
        family_id = str(family_row["family_id"])
        family = family_map[family_id]
        variant_rows = variant_rows_by_family[family_id]
        failing_variant_id = _primary_failure_variant(str(spec["task_family"]), variant_rows)
        primary_variant = variant_rows[failing_variant_id]
        primary_label, notes = _derive_primary_label(
            task_family=str(spec["task_family"]),
            family=family,
            primary_variant=primary_variant,
        )
        audit_rows.append(
            {
                "audit_id": f"{spec['run_id']}__{family_id}__{failing_variant_id}",
                "run_id": spec["run_id"],
                "task_family": spec["task_family"],
                "model_id": spec["model_id"],
                "prompt_id": spec["prompt_id"],
                "split": spec["split"],
                "family_id": family_id,
                "sample_seed": int(sample_seed),
                "sample_reason": "post_run_failure_gcf0_sample",
                "base_correct": int(family_row["base_correct"]),
                "para_1_or_lex_correct": int(
                    variant_rows[paraphrase_ids[0]]["pred_winner_cid"]
                    == variant_rows[paraphrase_ids[0]]["gold_winner_cid"]
                ),
                "para_2_or_struct_correct": int(
                    variant_rows[paraphrase_ids[1]]["pred_winner_cid"]
                    == variant_rows[paraphrase_ids[1]]["gold_winner_cid"]
                ),
                "counterfactual_correct": int(family_row["counterfactual_correct"]),
                "base_tie": int(variant_rows["base"]["pred_tie"]),
                "para_1_or_lex_tie": int(variant_rows[paraphrase_ids[0]]["pred_tie"]),
                "para_2_or_struct_tie": int(variant_rows[paraphrase_ids[1]]["pred_tie"]),
                "counterfactual_tie": int(variant_rows["counterfactual"]["pred_tie"]),
                "base_order_disagree": int(variant_rows["base"]["order_disagree"]),
                "para_1_or_lex_order_disagree": int(
                    variant_rows[paraphrase_ids[0]]["order_disagree"]
                ),
                "para_2_or_struct_order_disagree": int(
                    variant_rows[paraphrase_ids[1]]["order_disagree"]
                ),
                "counterfactual_order_disagree": int(
                    variant_rows["counterfactual"]["order_disagree"]
                ),
                "primary_failure_variant": failing_variant_id,
                "pred_base_cid": variant_rows["base"]["pred_winner_cid"],
                "pred_para_1_or_lex_cid": variant_rows[paraphrase_ids[0]]["pred_winner_cid"],
                "pred_para_2_or_struct_cid": variant_rows[paraphrase_ids[1]]["pred_winner_cid"],
                "pred_counterfactual_cid": variant_rows["counterfactual"]["pred_winner_cid"],
                "gold_base_cid": variant_rows["base"]["gold_winner_cid"],
                "gold_counterfactual_cid": variant_rows["counterfactual"]["gold_winner_cid"],
                "audit_complete": 1,
                "primary_label": primary_label,
                "notes": notes,
                "annotator_id": "manual_review",
                "timestamp_utc": timestamp_utc,
            }
        )

    csv_path = store.audit_dir / "failure_audit_sample.csv"
    summary_path = store.audit_dir / "failure_audit_summary.json"
    _write_csv(csv_path, audit_rows, fieldnames)
    atomic_write_json(
        summary_path,
        {
            "run_id": spec["run_id"],
            "task_family": spec["task_family"],
            "audit_type": "post_run_failure_summary",
            "sample_seed": int(sample_seed),
            "target_failures_if_available": int(target_failures),
            "n_failed_families_available": n_failed_families_available,
            "n_audit_complete": len(audit_rows),
            "label_counts": dict(
                sorted(Counter(row["primary_label"] for row in audit_rows).items())
            ),
            "all_available_failures_audited": n_failed_families_available <= int(target_failures),
            "csv_path": str(csv_path),
        },
    )
    return {
        "audit_csv": str(csv_path),
        "audit_summary_json": str(summary_path),
    }
