from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from src.data.constants import TIE_EPSILON
from src.eval.metrics import compute_family_result
from src.inference.score_ab import (
    LABELS,
    predict_winner_from_scores,
    remap_label_scores_to_candidates,
    render_variant_prefix,
)

ProgressCallback = Callable[[str, str, str, int], None]


def _display_order_candidates(order: str) -> tuple[str, str]:
    if order == "AB":
        return ("c1", "c2")
    if order == "BA":
        return ("c2", "c1")
    raise ValueError(f"Unknown order: {order}")


def _predict_display_label(label_scores: Mapping[str, float], eps: float) -> str:
    score_a = float(label_scores["A"])
    score_b = float(label_scores["B"])
    if abs(score_a - score_b) <= eps:
        return "TIE"
    if score_a > score_b:
        return "A"
    return "B"


def _prompt_record(
    *,
    family_id: str,
    variant_id: str,
    order: str,
    label_scores: Mapping[str, float],
    label_token_ids: Mapping[str, list[int]],
    label_token_logprobs: Mapping[str, list[float]],
    rendered_prefix: str,
    inference_ms: int,
    eps: float,
) -> dict[str, Any]:
    import hashlib

    displayed_cid_a, displayed_cid_b = _display_order_candidates(order)
    return {
        "family_id": family_id,
        "variant_id": variant_id,
        "order_id": order,
        "displayed_cid_a": displayed_cid_a,
        "displayed_cid_b": displayed_cid_b,
        "label_text_a": "A",
        "label_text_b": "B",
        "label_token_ids_a": list(label_token_ids["A"]),
        "label_token_ids_b": list(label_token_ids["B"]),
        "label_token_logprobs_a": [float(value) for value in label_token_logprobs["A"]],
        "label_token_logprobs_b": [float(value) for value in label_token_logprobs["B"]],
        "logprob_total_a": float(label_scores["A"]),
        "logprob_total_b": float(label_scores["B"]),
        "pred_display_label": _predict_display_label(label_scores, eps),
        "score_gap": float(label_scores["A"]) - float(label_scores["B"]),
        "rendered_prefix_text": rendered_prefix,
        "rendered_prefix_sha256": hashlib.sha256(rendered_prefix.encode("utf-8")).hexdigest(),
        "rendered_prefix_char_len": len(rendered_prefix),
        "inference_ms": int(inference_ms),
    }


def _variant_record(variant_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family_id": variant_result["family_id"],
        "variant_id": variant_result["variant_id"],
        "scores_ab_c1": float(variant_result["scores_ab"]["c1"]),
        "scores_ab_c2": float(variant_result["scores_ab"]["c2"]),
        "scores_ba_c1": float(variant_result["scores_ba"]["c1"]),
        "scores_ba_c2": float(variant_result["scores_ba"]["c2"]),
        "scores_agg_c1": float(variant_result["scores_agg"]["c1"]),
        "scores_agg_c2": float(variant_result["scores_agg"]["c2"]),
        "pred_winner_cid": str(variant_result["pred_winner_cid"]),
        "pred_tie": int(bool(variant_result["pred_tie"])),
        "gold_winner_cid": str(variant_result["gold_winner_cid"]),
        "order_pred_ab": str(variant_result["order_pred_ab"]),
        "order_pred_ba": str(variant_result["order_pred_ba"]),
        "order_tie_ab": int(bool(variant_result["order_tie_ab"])),
        "order_tie_ba": int(bool(variant_result["order_tie_ba"])),
        "order_disagree": int(bool(variant_result["order_disagree"])),
    }


def score_family(
    *,
    family: Mapping[str, Any],
    tokenizer: Any,
    scorer: Any,
    prompt_name: str,
    eps: float = TIE_EPSILON,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Score one family fully and return prompt-, variant-, and family-level artifacts."""

    variant_results: list[dict[str, Any]] = []
    prompt_results: list[dict[str, Any]] = []
    prompt_index = 0

    for variant in family["variants"]:
        if progress_callback is not None:
            progress_callback(
                str(family["family_id"]), str(variant["variant_id"]), "AB", prompt_index
            )

        prefix_ab = render_variant_prefix(
            family=family,
            variant=variant,
            tokenizer=tokenizer,
            prompt_name=prompt_name,
            order="AB",
        )
        start_ab = time.perf_counter()
        scored_ab = scorer.score_labels(prefix_ab, LABELS)
        inference_ms_ab = int((time.perf_counter() - start_ab) * 1000)
        scores_ab = remap_label_scores_to_candidates(scored_ab.label_logprobs, order="AB")
        order_pred_ab, order_tie_ab = predict_winner_from_scores(scores_ab, eps)

        prompt_results.append(
            _prompt_record(
                family_id=str(family["family_id"]),
                variant_id=str(variant["variant_id"]),
                order="AB",
                label_scores=scored_ab.label_logprobs,
                label_token_ids=scored_ab.label_token_ids,
                label_token_logprobs=scored_ab.label_token_logprobs,
                rendered_prefix=prefix_ab,
                inference_ms=inference_ms_ab,
                eps=eps,
            )
        )
        prompt_index += 1

        if progress_callback is not None:
            progress_callback(
                str(family["family_id"]), str(variant["variant_id"]), "BA", prompt_index
            )

        prefix_ba = render_variant_prefix(
            family=family,
            variant=variant,
            tokenizer=tokenizer,
            prompt_name=prompt_name,
            order="BA",
        )
        start_ba = time.perf_counter()
        scored_ba = scorer.score_labels(prefix_ba, LABELS)
        inference_ms_ba = int((time.perf_counter() - start_ba) * 1000)
        scores_ba = remap_label_scores_to_candidates(scored_ba.label_logprobs, order="BA")
        order_pred_ba, order_tie_ba = predict_winner_from_scores(scores_ba, eps)

        prompt_results.append(
            _prompt_record(
                family_id=str(family["family_id"]),
                variant_id=str(variant["variant_id"]),
                order="BA",
                label_scores=scored_ba.label_logprobs,
                label_token_ids=scored_ba.label_token_ids,
                label_token_logprobs=scored_ba.label_token_logprobs,
                rendered_prefix=prefix_ba,
                inference_ms=inference_ms_ba,
                eps=eps,
            )
        )
        prompt_index += 1

        scores_agg = {
            "c1": 0.5 * (scores_ab["c1"] + scores_ba["c1"]),
            "c2": 0.5 * (scores_ab["c2"] + scores_ba["c2"]),
        }
        pred_winner_cid, pred_tie = predict_winner_from_scores(scores_agg, eps)
        variant_results.append(
            {
                "family_id": family["family_id"],
                "variant_id": variant["variant_id"],
                "scores_ab": scores_ab,
                "scores_ba": scores_ba,
                "scores_agg": scores_agg,
                "pred_winner_cid": pred_winner_cid,
                "pred_tie": pred_tie,
                "gold_winner_cid": variant["gold_winner_cid"],
                "order_pred_ab": order_pred_ab,
                "order_pred_ba": order_pred_ba,
                "order_tie_ab": order_tie_ab,
                "order_tie_ba": order_tie_ba,
                "order_disagree": order_pred_ab != order_pred_ba,
                "label_scores_ab": dict(scored_ab.label_logprobs),
                "label_scores_ba": dict(scored_ba.label_logprobs),
                "label_token_ids": dict(scored_ab.label_token_ids),
                "label_token_logprobs": dict(scored_ab.label_token_logprobs),
                "label_token_ids_ba": dict(scored_ba.label_token_ids),
                "label_token_logprobs_ba": dict(scored_ba.label_token_logprobs),
                "rendered_prefix_ab": prefix_ab,
                "rendered_prefix_ba": prefix_ba,
                "inference_ms_ab": inference_ms_ab,
                "inference_ms_ba": inference_ms_ba,
            }
        )

    family_result = compute_family_result(family, variant_results)
    return {
        "prompt_results": prompt_results,
        "variant_results": [_variant_record(variant_result) for variant_result in variant_results],
        "family_result": family_result,
    }
