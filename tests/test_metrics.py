from __future__ import annotations

from copy import deepcopy

from src.data.toy_records import TOY_QA_FAMILY
from src.eval.metrics import compute_family_result, compute_summary_metrics, run_self_test


def _variant_eval(variant_id: str, pred_winner_cid: str, gold_winner_cid: str) -> dict[str, object]:
    return {
        "family_id": "family_demo",
        "variant_id": variant_id,
        "scores_ab": {"c1": -0.1, "c2": -0.2},
        "scores_ba": {"c1": -0.1, "c2": -0.2},
        "scores_agg": {"c1": -0.1, "c2": -0.2},
        "pred_winner_cid": pred_winner_cid,
        "pred_tie": pred_winner_cid == "tie",
        "gold_winner_cid": gold_winner_cid,
        "order_pred_ab": pred_winner_cid,
        "order_pred_ba": pred_winner_cid,
        "order_tie_ab": pred_winner_cid == "tie",
        "order_tie_ba": pred_winner_cid == "tie",
        "order_disagree": False,
        "label_scores_ab": {"A": -0.1, "B": -0.2},
        "label_scores_ba": {"A": -0.1, "B": -0.2},
        "label_token_ids": {"A": [11], "B": [12]},
        "label_token_ids_ba": {"A": [11], "B": [12]},
        "rendered_prefix_ab": "prefix_ab",
        "rendered_prefix_ba": "prefix_ba",
    }


def test_metric_self_test_matches_locked_toy_table() -> None:
    summary = run_self_test()

    assert summary["base_acc"] == 2 / 3
    assert summary["gcf"] == 1 / 3
    assert summary["sensitivity_at_base_correct"] == 0.5
    assert summary["invariance_at_base_correct"] == 1.0


def test_gcf_never_exceeds_base_accuracy() -> None:
    summary = compute_summary_metrics(
        [
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
                "paraphrase_all_correct": 0,
                "counterfactual_correct": 1,
                "gcf_success": 0,
                "tie_count": 0,
                "variant_count": 4,
                "order_disagreement_count": 0,
                "variant_results": [],
            },
        ]
    )

    assert summary["gcf"] <= summary["base_acc"]


def test_tied_counterfactual_counts_incorrect_for_family_metrics() -> None:
    family = deepcopy(TOY_QA_FAMILY)
    family_result = compute_family_result(
        family,
        [
            _variant_eval("base", "c1", "c1"),
            _variant_eval("para_1", "c1", "c1"),
            _variant_eval("para_2", "c1", "c1"),
            _variant_eval("counterfactual", "tie", "c2"),
        ],
    )

    assert family_result["base_correct"] == 1
    assert family_result["paraphrase_all_correct"] == 1
    assert family_result["counterfactual_correct"] == 0
    assert family_result["gcf_success"] == 0
    assert family_result["tie_count"] == 1
