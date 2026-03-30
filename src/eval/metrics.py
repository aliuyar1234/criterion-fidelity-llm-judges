from __future__ import annotations

import argparse
import json
from typing import Any, Iterable, Mapping

from src.data.constants import TASK_FAMILY_PARAPHRASE_IDS, TASK_FAMILY_VARIANT_IDS


def _mean(values: Iterable[int | float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        raise ValueError("Cannot compute a mean over an empty iterable.")
    return sum(values_list) / len(values_list)


def compute_family_result(
    family: Mapping[str, Any],
    variant_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate logical-variant predictions into one family-level result."""

    task_family = str(family["task_family"])
    expected_variant_ids = set(TASK_FAMILY_VARIANT_IDS[task_family])
    paraphrase_ids = set(TASK_FAMILY_PARAPHRASE_IDS[task_family])
    variant_results_by_id = {
        str(variant_result["variant_id"]): dict(variant_result)
        for variant_result in variant_results
    }

    if set(variant_results_by_id) != expected_variant_ids:
        raise ValueError(
            f"Expected variant results for {sorted(expected_variant_ids)}, "
            f"got {sorted(variant_results_by_id)}."
        )

    base_correct = int(
        variant_results_by_id["base"]["pred_winner_cid"]
        == variant_results_by_id["base"]["gold_winner_cid"]
    )
    paraphrase_all_correct = int(
        all(
            variant_results_by_id[variant_id]["pred_winner_cid"]
            == variant_results_by_id[variant_id]["gold_winner_cid"]
            for variant_id in paraphrase_ids
        )
    )
    counterfactual_correct = int(
        variant_results_by_id["counterfactual"]["pred_winner_cid"]
        == variant_results_by_id["counterfactual"]["gold_winner_cid"]
    )
    gcf_success = base_correct * paraphrase_all_correct * counterfactual_correct
    tie_count = sum(
        int(bool(variant_result["pred_tie"])) for variant_result in variant_results_by_id.values()
    )
    order_disagreement_count = sum(
        int(bool(variant_result["order_disagree"]))
        for variant_result in variant_results_by_id.values()
    )

    return {
        "family_id": family["family_id"],
        "task_family": task_family,
        "split": family["split"],
        "base_correct": base_correct,
        "paraphrase_all_correct": paraphrase_all_correct,
        "counterfactual_correct": counterfactual_correct,
        "gcf_success": gcf_success,
        "tie_count": tie_count,
        "variant_count": len(variant_results_by_id),
        "order_disagreement_count": order_disagreement_count,
        "variant_results": [
            variant_results_by_id[variant_id] for variant_id in TASK_FAMILY_VARIANT_IDS[task_family]
        ],
    }


def compute_summary_metrics(family_results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the locked dataset-level headline and supporting metrics."""

    family_results_list = [dict(family_result) for family_result in family_results]
    if not family_results_list:
        raise ValueError("At least one family result is required to compute summary metrics.")

    n_families = len(family_results_list)
    n_base_correct = sum(
        int(family_result["base_correct"]) for family_result in family_results_list
    )
    total_variant_count = sum(
        int(family_result["variant_count"]) for family_result in family_results_list
    )
    total_ties = sum(int(family_result["tie_count"]) for family_result in family_results_list)
    total_order_disagreements = sum(
        int(family_result["order_disagreement_count"]) for family_result in family_results_list
    )

    base_acc = _mean(int(family_result["base_correct"]) for family_result in family_results_list)
    gcf = _mean(int(family_result["gcf_success"]) for family_result in family_results_list)

    if n_base_correct == 0:
        sens_at_base_correct = None
        inv_at_base_correct = None
        underreaction_at_base_correct = None
        overreaction_at_base_correct = None
    else:
        sens_at_base_correct = (
            sum(
                int(family_result["base_correct"]) * int(family_result["counterfactual_correct"])
                for family_result in family_results_list
            )
            / n_base_correct
        )
        inv_at_base_correct = (
            sum(
                int(family_result["base_correct"]) * int(family_result["paraphrase_all_correct"])
                for family_result in family_results_list
            )
            / n_base_correct
        )
        underreaction_at_base_correct = 1.0 - sens_at_base_correct
        overreaction_at_base_correct = 1.0 - inv_at_base_correct

    return {
        "n_families": n_families,
        "n_base_correct": n_base_correct,
        "n_logical_variants": total_variant_count,
        "base_acc": base_acc,
        "gcf": gcf,
        "sensitivity_at_base_correct": sens_at_base_correct,
        "invariance_at_base_correct": inv_at_base_correct,
        "underreaction_at_base_correct": underreaction_at_base_correct,
        "overreaction_at_base_correct": overreaction_at_base_correct,
        "gap": base_acc - gcf,
        "tie_rate": total_ties / total_variant_count,
        "order_disagreement": total_order_disagreements / total_variant_count,
    }


def _assert_close(actual: float | None, expected: float | None, name: str) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise AssertionError(f"{name} mismatch: expected {expected!r}, got {actual!r}")
        return
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"{name} mismatch: expected {expected}, got {actual}")


def _build_self_test_family_results() -> list[dict[str, Any]]:
    return [
        {
            "family_id": "F1",
            "task_family": "qa_key",
            "split": "dev",
            "base_correct": 1,
            "paraphrase_all_correct": 1,
            "counterfactual_correct": 1,
            "gcf_success": 1,
            "tie_count": 0,
            "variant_count": 4,
            "order_disagreement_count": 0,
            "variant_results": [],
        },
        {
            "family_id": "F2",
            "task_family": "qa_key",
            "split": "dev",
            "base_correct": 1,
            "paraphrase_all_correct": 1,
            "counterfactual_correct": 0,
            "gcf_success": 0,
            "tie_count": 1,
            "variant_count": 4,
            "order_disagreement_count": 1,
            "variant_results": [],
        },
        {
            "family_id": "F3",
            "task_family": "qa_key",
            "split": "dev",
            "base_correct": 0,
            "paraphrase_all_correct": 1,
            "counterfactual_correct": 1,
            "gcf_success": 0,
            "tie_count": 0,
            "variant_count": 4,
            "order_disagreement_count": 0,
            "variant_results": [],
        },
    ]


def run_self_test() -> dict[str, Any]:
    """Validate the toy metric table from the locked hard-parts document."""

    summary = compute_summary_metrics(_build_self_test_family_results())

    _assert_close(summary["base_acc"], 2 / 3, "base_acc")
    _assert_close(summary["gcf"], 1 / 3, "gcf")
    _assert_close(summary["sensitivity_at_base_correct"], 0.5, "sensitivity_at_base_correct")
    _assert_close(summary["invariance_at_base_correct"], 1.0, "invariance_at_base_correct")
    _assert_close(summary["underreaction_at_base_correct"], 0.5, "underreaction_at_base_correct")
    _assert_close(summary["overreaction_at_base_correct"], 0.0, "overreaction_at_base_correct")

    if summary["gcf"] > summary["base_acc"]:
        raise AssertionError("GCF must never exceed Base Accuracy.")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Family-level metric utilities.")
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run the locked toy metric checks.",
    )
    args = parser.parse_args()

    if not args.self_test:
        parser.error("No action specified. Use --self_test.")

    summary = run_self_test()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
