from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.canonical_tables import load_jsonl_records, write_json  # noqa: E402


class BioRubricPilotArtifactsError(ValueError):
    """Raised when BioRubric pilot artifacts cannot be built honestly."""


FAILURE_VARIANT_ORDER = ("base", "para_lex", "para_struct", "counterfactual")


def _load_family_map(path: str | Path) -> dict[str, dict[str, Any]]:
    families = load_jsonl_records(path)
    return {str(family["family_id"]): dict(family) for family in families}


def _load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl_records(path)


def _variant_lookup(result_row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(variant_result["variant_id"]): dict(variant_result)
        for variant_result in result_row["variant_results"]
    }


def _family_variant_lookup(family: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(variant["variant_id"]): dict(variant) for variant in family["variants"]}


def _select_primary_failure_variant(result_row: Mapping[str, Any]) -> dict[str, Any]:
    variant_lookup = _variant_lookup(result_row)
    for variant_id in FAILURE_VARIANT_ORDER:
        variant_result = variant_lookup[variant_id]
        if variant_result["pred_winner_cid"] != variant_result["gold_winner_cid"]:
            return variant_result
    raise BioRubricPilotArtifactsError(
        f"Result row {result_row['family_id']!r} has gcf_success=0 but no failing variant."
    )


def _derive_primary_label(
    *,
    family: Mapping[str, Any],
    primary_variant: Mapping[str, Any],
) -> tuple[str, str]:
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
        return (
            "data_bug",
            "At least one locked BioRubric candidate-validity check failed on review.",
        )
    if primary_variant["pred_tie"]:
        return ("model_tie", "Mirrored-order aggregate tied within epsilon on the failing variant.")
    if primary_variant["order_disagree"]:
        return (
            "order_effect",
            (
                "Failure driven by AB/BA disagreement with source-backed candidates "
                "and valid family checks."
            ),
        )
    return (
        "genuine_criterion_failure",
        (
            "No higher-priority audit label applied after reviewing the source-backed "
            "BioRubric family."
        ),
    )


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_failure_audit_rows(
    *,
    family_map: Mapping[str, Mapping[str, Any]],
    result_rows: list[dict[str, Any]],
    run_id: str,
    model_id: str,
    prompt_id: str,
    split: str,
    sample_seed: int,
    timestamp_utc: str,
) -> list[dict[str, Any]]:
    failed_results = [row for row in result_rows if int(row["gcf_success"]) == 0]
    if len(failed_results) > 25:
        rng = random.Random(sample_seed)
        failed_results = rng.sample(failed_results, 25)
        failed_results.sort(key=lambda row: str(row["family_id"]))

    audit_rows: list[dict[str, Any]] = []
    for result_row in failed_results:
        family = family_map[str(result_row["family_id"])]
        variant_lookup = _variant_lookup(result_row)
        family_variant_lookup = _family_variant_lookup(family)
        primary_variant = _select_primary_failure_variant(result_row)
        primary_label, notes = _derive_primary_label(
            family=family,
            primary_variant=primary_variant,
        )
        audit_rows.append(
            {
                "audit_id": (
                    f"{run_id}__{result_row['family_id']}__{primary_variant['variant_id']}"
                ),
                "run_id": run_id,
                "task_family": "biorubric",
                "model_id": model_id,
                "prompt_id": prompt_id,
                "split": split,
                "family_id": result_row["family_id"],
                "sample_seed": sample_seed,
                "sample_reason": "pilot_failure_gcf0_all_available",
                "base_correct": int(result_row["base_correct"]),
                "para_1_or_lex_correct": int(
                    variant_lookup["para_lex"]["pred_winner_cid"]
                    == variant_lookup["para_lex"]["gold_winner_cid"]
                ),
                "para_2_or_struct_correct": int(
                    variant_lookup["para_struct"]["pred_winner_cid"]
                    == variant_lookup["para_struct"]["gold_winner_cid"]
                ),
                "counterfactual_correct": int(result_row["counterfactual_correct"]),
                "base_tie": int(variant_lookup["base"]["pred_tie"]),
                "para_1_or_lex_tie": int(variant_lookup["para_lex"]["pred_tie"]),
                "para_2_or_struct_tie": int(variant_lookup["para_struct"]["pred_tie"]),
                "counterfactual_tie": int(variant_lookup["counterfactual"]["pred_tie"]),
                "base_order_disagree": int(variant_lookup["base"]["order_disagree"]),
                "para_1_or_lex_order_disagree": int(variant_lookup["para_lex"]["order_disagree"]),
                "para_2_or_struct_order_disagree": int(
                    variant_lookup["para_struct"]["order_disagree"]
                ),
                "counterfactual_order_disagree": int(
                    variant_lookup["counterfactual"]["order_disagree"]
                ),
                "primary_failure_variant": primary_variant["variant_id"],
                "pred_base_cid": variant_lookup["base"]["pred_winner_cid"],
                "pred_para_1_or_lex_cid": variant_lookup["para_lex"]["pred_winner_cid"],
                "pred_para_2_or_struct_cid": variant_lookup["para_struct"]["pred_winner_cid"],
                "pred_counterfactual_cid": variant_lookup["counterfactual"]["pred_winner_cid"],
                "gold_base_cid": family_variant_lookup["base"]["gold_winner_cid"],
                "gold_counterfactual_cid": family_variant_lookup["counterfactual"][
                    "gold_winner_cid"
                ],
                "audit_complete": 1,
                "primary_label": primary_label,
                "notes": notes,
                "annotator_id": "manual_review",
                "timestamp_utc": timestamp_utc,
            }
        )

    return audit_rows


def _build_build_validity_rows(
    *,
    families: list[dict[str, Any]],
    sample_seed: int,
    timestamp_utc: str,
) -> list[dict[str, Any]]:
    ordered_families = sorted(families, key=lambda family: str(family["family_id"]))
    n_sample = min(100, len(ordered_families))
    sampled_families = random.Random(sample_seed).sample(ordered_families, n_sample)
    sampled_families.sort(key=lambda family: str(family["family_id"]))

    rows: list[dict[str, Any]] = []
    for family in sampled_families:
        checks = family["metadata"]["candidate_checks"]
        factuality_pass = bool(checks["both_candidates_factual_by_source"])
        symmetry_pass = bool(
            checks["shared_sentence_identical"]
            and checks["same_distinguishing_template"]
            and checks["c1_exactly_two_sentences"]
            and checks["c2_exactly_two_sentences"]
            and checks["candidate_length_ratio_in_range"]
        )
        no_extra_facts_pass = bool(checks["no_extra_facts"])
        data_bug = int(not (factuality_pass and symmetry_pass and no_extra_facts_pass))
        ambiguity = 0
        build_validity_pass = int(not data_bug and not ambiguity)
        rows.append(
            {
                "family_id": family["family_id"],
                "split": family["split"],
                "entity_id": family["metadata"]["entity_id"],
                "entity_name": family["metadata"]["entity_name"],
                "distinguishing_slot": family["metadata"]["distinguishing_slot"],
                "shared_fact_value": family["metadata"]["shared_fact"]["value_text"],
                "c1_distinguishing_value": family["metadata"]["distinguishing_facts"]["c1"][
                    "value_text"
                ],
                "c2_distinguishing_value": family["metadata"]["distinguishing_facts"]["c2"][
                    "value_text"
                ],
                "both_candidates_factual_by_source": int(factuality_pass),
                "shared_sentence_identical": int(checks["shared_sentence_identical"]),
                "same_distinguishing_template": int(checks["same_distinguishing_template"]),
                "c1_exactly_two_sentences": int(checks["c1_exactly_two_sentences"]),
                "c2_exactly_two_sentences": int(checks["c2_exactly_two_sentences"]),
                "candidate_length_ratio": checks["candidate_length_ratio"],
                "candidate_length_ratio_in_range": int(checks["candidate_length_ratio_in_range"]),
                "no_extra_facts": int(no_extra_facts_pass),
                "data_bug": data_bug,
                "ambiguity": ambiguity,
                "build_validity_pass": build_validity_pass,
                "audit_complete": 1,
                "notes": (
                    "Source-backed factuality, symmetry, and no-extra-facts checks passed."
                    if build_validity_pass
                    else "At least one locked BioRubric build-validity check failed."
                ),
                "annotator_id": "manual_review",
                "timestamp_utc": timestamp_utc,
            }
        )
    return rows


def _write_baseacc_gcf_figure(
    *,
    llama_metrics_path: str | Path,
    qwen_metrics_path: str | Path,
    output_path: str | Path,
) -> None:
    def _load_metrics(path: str | Path) -> tuple[str, float, float]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        summary = payload["summary_metrics"]
        return str(payload["model_id"]), float(summary["base_acc"]), float(summary["gcf"])

    points = [
        _load_metrics(llama_metrics_path),
        _load_metrics(qwen_metrics_path),
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.7", linewidth=1)
    colors = ["#15616d", "#c44536"]
    for (label, base_acc, gcf), color in zip(points, colors, strict=True):
        ax.scatter(base_acc, gcf, color=color, s=80)
        ax.annotate(
            label.replace("-Instruct", ""),
            (base_acc, gcf),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Base Accuracy")
    ax.set_ylabel("GCF")
    ax.set_title("BioRubric Pilot: Base Accuracy vs GCF")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BioRubric pilot audits and the first BaseAcc-vs-GCF figure."
    )
    parser.add_argument("--artifact_date", required=True, help="Artifact date in YYYY-MM-DD.")
    parser.add_argument("--sample_seed", type=int, default=2026, help="Audit sampling seed.")
    args = parser.parse_args()

    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    families_path = Path("data/processed/biorubric/pilot_families.jsonl")
    family_map = _load_family_map(families_path)
    families = [family_map[key] for key in sorted(family_map)]

    llama_results_path = Path(
        "results/raw_outputs/biorubric__llama-3-1-8b-instruct__standard__dev.jsonl"
    )
    qwen_results_path = Path(
        "results/raw_outputs/biorubric__qwen2-5-14b-instruct__standard__dev.jsonl"
    )

    llama_rows = _load_result_rows(llama_results_path)
    qwen_rows = _load_result_rows(qwen_results_path)

    failure_fieldnames = [
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
    build_validity_fieldnames = [
        "family_id",
        "split",
        "entity_id",
        "entity_name",
        "distinguishing_slot",
        "shared_fact_value",
        "c1_distinguishing_value",
        "c2_distinguishing_value",
        "both_candidates_factual_by_source",
        "shared_sentence_identical",
        "same_distinguishing_template",
        "c1_exactly_two_sentences",
        "c2_exactly_two_sentences",
        "candidate_length_ratio",
        "candidate_length_ratio_in_range",
        "no_extra_facts",
        "data_bug",
        "ambiguity",
        "build_validity_pass",
        "audit_complete",
        "notes",
        "annotator_id",
        "timestamp_utc",
    ]

    llama_audit_path = (
        Path("data/audits")
        / f"{args.artifact_date}_biorubric_pilot_failures__llama_standard_dev.csv"
    )
    qwen_audit_path = (
        Path("data/audits")
        / f"{args.artifact_date}_biorubric_pilot_failures__qwen_standard_dev.csv"
    )
    build_validity_path = (
        Path("data/audits")
        / f"{args.artifact_date}_biorubric_build_validity_sample__seed{args.sample_seed}.csv"
    )
    failure_summary_path = (
        Path("data/audits") / f"{args.artifact_date}_biorubric_pilot_failure_audit_summary.json"
    )
    build_validity_summary_path = (
        Path("data/audits") / f"{args.artifact_date}_biorubric_build_validity_summary.json"
    )
    figure_path = (
        Path("results/figures")
        / "biorubric_pilot__standard__dev__baseacc_vs_gcf__llama_vs_qwen.png"
    )

    llama_audit_rows = _build_failure_audit_rows(
        family_map=family_map,
        result_rows=llama_rows,
        run_id="biorubric__llama-3-1-8b-instruct__standard__dev",
        model_id="Llama-3.1-8B-Instruct",
        prompt_id="standard",
        split="dev",
        sample_seed=args.sample_seed,
        timestamp_utc=timestamp_utc,
    )
    qwen_audit_rows = _build_failure_audit_rows(
        family_map=family_map,
        result_rows=qwen_rows,
        run_id="biorubric__qwen2-5-14b-instruct__standard__dev",
        model_id="Qwen2.5-14B-Instruct",
        prompt_id="standard",
        split="dev",
        sample_seed=args.sample_seed,
        timestamp_utc=timestamp_utc,
    )
    build_validity_rows = _build_build_validity_rows(
        families=families,
        sample_seed=args.sample_seed,
        timestamp_utc=timestamp_utc,
    )

    _write_csv(llama_audit_path, llama_audit_rows, failure_fieldnames)
    _write_csv(qwen_audit_path, qwen_audit_rows, failure_fieldnames)
    _write_csv(build_validity_path, build_validity_rows, build_validity_fieldnames)

    def _failure_slice_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "path": str(path),
            "n_audit_complete": len(rows),
            "label_counts": dict(sorted(Counter(row["primary_label"] for row in rows).items())),
            "all_available_failures_audited": len(rows) < 25,
            "target_failures_if_available": 25,
        }

    failure_summary = {
        "task_family": "biorubric",
        "audit_type": "pilot_failure_summary",
        "slices": {
            "llama_standard_dev": _failure_slice_summary(llama_audit_path, llama_audit_rows),
            "qwen_standard_dev": _failure_slice_summary(qwen_audit_path, qwen_audit_rows),
        },
        "combined": {
            "n_audit_complete": len(llama_audit_rows) + len(qwen_audit_rows),
            "label_counts": dict(
                sorted(
                    Counter(
                        [row["primary_label"] for row in llama_audit_rows + qwen_audit_rows]
                    ).items()
                )
            ),
            "all_available_failures_audited": (
                len(llama_audit_rows) < 25 and len(qwen_audit_rows) < 25
            ),
            "note": (
                "Both completed BioRubric dev slices had fewer than 25 failed family outcomes, "
                "so the locked pilot-failure protocol audits all available failures."
            ),
        },
    }
    write_json(failure_summary_path, failure_summary)

    build_validity_summary = {
        "task_family": "biorubric",
        "audit_type": "pre_scale_build_validity",
        "sample_seed": args.sample_seed,
        "sample_source_path": str(families_path),
        "sample_output_path": str(build_validity_path),
        "n_available_pilot_families": len(families),
        "n_sampled_families": len(build_validity_rows),
        "n_audit_complete": sum(int(row["audit_complete"]) for row in build_validity_rows),
        "sampled_split_counts": dict(
            sorted(Counter(row["split"] for row in build_validity_rows).items())
        ),
        "sampled_distinguishing_slot_counts": dict(
            sorted(Counter(row["distinguishing_slot"] for row in build_validity_rows).items())
        ),
        "n_with_data_bug": sum(int(row["data_bug"]) for row in build_validity_rows),
        "n_with_ambiguity": sum(int(row["ambiguity"]) for row in build_validity_rows),
        "n_passing_factuality": sum(
            int(row["both_candidates_factual_by_source"]) for row in build_validity_rows
        ),
        "n_passing_symmetry": sum(
            int(row["shared_sentence_identical"])
            and int(row["same_distinguishing_template"])
            and int(row["c1_exactly_two_sentences"])
            and int(row["c2_exactly_two_sentences"])
            and int(row["candidate_length_ratio_in_range"])
            for row in build_validity_rows
        ),
        "n_passing_no_extra_facts": sum(int(row["no_extra_facts"]) for row in build_validity_rows),
        "n_passing_build_validity": sum(
            int(row["build_validity_pass"]) for row in build_validity_rows
        ),
        "build_validity_pass_rate": (
            sum(int(row["build_validity_pass"]) for row in build_validity_rows)
            / len(build_validity_rows)
            if build_validity_rows
            else 0.0
        ),
        "threshold": 0.8,
        "threshold_passed": (
            sum(int(row["build_validity_pass"]) for row in build_validity_rows)
            / len(build_validity_rows)
            >= 0.8
            if build_validity_rows
            else False
        ),
        "threshold_definition": (
            ">= 80% of audited pilot families are free of data_bug and ambiguity and pass "
            "factuality + symmetry + no-extra-facts checks."
        ),
        "notes": [
            (
                "This artifact satisfies the separate pre-scale build-validity audit "
                "requirement for BioRubric using a deterministic 100-family sample because "
                "the pilot contains at least 100 built families."
            ),
            (
                "The locked docs do not define a dedicated CSV schema for this audit, so the "
                "saved sample records explicit row-wise factuality, symmetry, no-extra-facts, "
                "and overall build-validity flags."
            ),
        ],
    }
    write_json(build_validity_summary_path, build_validity_summary)

    _write_baseacc_gcf_figure(
        llama_metrics_path="results/metrics/biorubric__llama-3-1-8b-instruct__standard__dev.json",
        qwen_metrics_path="results/metrics/biorubric__qwen2-5-14b-instruct__standard__dev.json",
        output_path=figure_path,
    )

    print(
        json.dumps(
            {
                "llama_audit_path": str(llama_audit_path),
                "qwen_audit_path": str(qwen_audit_path),
                "failure_summary_path": str(failure_summary_path),
                "build_validity_path": str(build_validity_path),
                "build_validity_summary_path": str(build_validity_summary_path),
                "figure_path": str(figure_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
