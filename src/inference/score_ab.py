from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch

from src.data.constants import TIE_EPSILON
from src.inference.prompts import build_messages, render_chat_prefix

LABELS = ("A", "B")
ORDERS = ("AB", "BA")
PRIMARY_MODEL_REPO_IDS = {
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


@dataclass(frozen=True)
class ScoredLabels:
    """Score totals and token IDs for a fixed prefix over a small label set."""

    label_logprobs: dict[str, float]
    label_token_ids: dict[str, list[int]]
    label_token_logprobs: dict[str, list[float]]


class ContinuationScorer(Protocol):
    """Protocol for scoring continuation strings from an identical prefix."""

    def score_labels(self, prefix: str, labels: Sequence[str] = LABELS) -> ScoredLabels:
        """Return summed continuation logprobs and the token IDs actually scored."""


def _candidate_text_map(family: Mapping[str, Any]) -> dict[str, str]:
    return {candidate["cid"]: candidate["text"] for candidate in family["candidates"]}


def render_variant_prefix(
    family: Mapping[str, Any],
    variant: Mapping[str, Any],
    tokenizer: Any,
    prompt_name: str,
    order: str,
) -> str:
    """Render one order-specific chat prefix that ends before the final label."""

    candidate_texts = _candidate_text_map(family)
    if order == "AB":
        candidate_a_text = candidate_texts["c1"]
        candidate_b_text = candidate_texts["c2"]
    elif order == "BA":
        candidate_a_text = candidate_texts["c2"]
        candidate_b_text = candidate_texts["c1"]
    else:
        raise ValueError(f"Unknown order: {order}")

    messages = build_messages(
        prompt_name=prompt_name,
        task_text=family["task_text"],
        criterion_text=variant["criterion_text"],
        candidate_a_text=candidate_a_text,
        candidate_b_text=candidate_b_text,
    )
    return render_chat_prefix(tokenizer, messages)


def predict_winner_from_scores(
    scores_by_candidate: Mapping[str, float],
    eps: float,
) -> tuple[str, bool]:
    """Apply the locked tie rule to a `{c1, c2}` score mapping."""

    c1_score = float(scores_by_candidate["c1"])
    c2_score = float(scores_by_candidate["c2"])
    if abs(c1_score - c2_score) <= eps:
        return "tie", True
    if c1_score > c2_score:
        return "c1", False
    return "c2", False


def remap_label_scores_to_candidates(
    label_scores: Mapping[str, float],
    order: str,
) -> dict[str, float]:
    """Map order-specific label scores back to the underlying candidate IDs."""

    if order == "AB":
        return {"c1": float(label_scores["A"]), "c2": float(label_scores["B"])}
    if order == "BA":
        return {"c1": float(label_scores["B"]), "c2": float(label_scores["A"])}
    raise ValueError(f"Unknown order: {order}")


def _tokenize_text(tokenizer: Any, text: str) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=getattr(tokenizer, "is_fast", False),
    )
    if not isinstance(encoded["input_ids"], list):
        raise ValueError("Tokenizer returned an unexpected input_ids format.")
    return encoded


def extract_continuation_token_ids(
    tokenizer: Any,
    prefix: str,
    label: str,
) -> tuple[list[int], list[int], list[int]]:
    """Return full IDs, continuation positions, and continuation token IDs."""

    full_encoded = _tokenize_text(tokenizer, prefix + label)
    full_ids = [int(token_id) for token_id in full_encoded["input_ids"]]
    boundary = len(prefix)

    if "offset_mapping" in full_encoded:
        offsets = full_encoded["offset_mapping"]
        for start, end in offsets:
            if start < boundary < end:
                raise ValueError(
                    "Tokenizer produced a token that crosses the prefix/continuation boundary."
                )
        continuation_positions = [
            index for index, (start, _) in enumerate(offsets) if start >= boundary
        ]
    else:
        prefix_ids = _tokenize_text(tokenizer, prefix)["input_ids"]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Cannot isolate continuation token IDs without offset mappings for this tokenizer."
            )
        continuation_positions = list(range(len(prefix_ids), len(full_ids)))

    if not continuation_positions:
        raise ValueError("No continuation token IDs were found for the requested label.")

    continuation_ids = [full_ids[position] for position in continuation_positions]
    return full_ids, continuation_positions, continuation_ids


class TransformersContinuationScorer:
    """Continuation scorer backed by a causal LM and tokenizer."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        self.model.eval()

    def score_labels(self, prefix: str, labels: Sequence[str] = LABELS) -> ScoredLabels:
        label_logprobs: dict[str, float] = {}
        label_token_ids: dict[str, list[int]] = {}
        label_token_logprobs: dict[str, list[float]] = {}

        encoded_labels: list[tuple[str, list[int], list[int], list[int]]] = []
        for label in labels:
            full_ids, continuation_positions, continuation_ids = extract_continuation_token_ids(
                self.tokenizer,
                prefix=prefix,
                label=label,
            )
            label_token_ids[str(label)] = continuation_ids
            encoded_labels.append((str(label), full_ids, continuation_positions, continuation_ids))

        with torch.inference_mode():
            max_length = max(len(full_ids) for _, full_ids, _, _ in encoded_labels)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0

            input_ids = torch.full(
                (len(encoded_labels), max_length),
                fill_value=int(pad_token_id),
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)

            for row_index, (_, full_ids, _, _) in enumerate(encoded_labels):
                input_ids[row_index, : len(full_ids)] = torch.tensor(
                    full_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask[row_index, : len(full_ids)] = 1

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

            for row_index, (label, full_ids, continuation_positions, _) in enumerate(
                encoded_labels
            ):
                continuation_score = 0.0
                token_logprobs: list[float] = []
                for position in continuation_positions:
                    if position == 0:
                        raise ValueError("Continuation scoring requires a non-empty prefix.")
                    token_id = full_ids[position]
                    token_logprob = float(log_probs[row_index, position - 1, token_id].item())
                    continuation_score += token_logprob
                    token_logprobs.append(token_logprob)

                label_logprobs[label] = continuation_score
                label_token_logprobs[label] = token_logprobs

        return ScoredLabels(
            label_logprobs=label_logprobs,
            label_token_ids=label_token_ids,
            label_token_logprobs=label_token_logprobs,
        )


class MockContinuationScorer:
    """Test double that returns precomputed scores keyed by rendered prefix."""

    def __init__(
        self,
        scores_by_prefix: Mapping[str, Mapping[str, float]],
        token_ids_by_prefix: Mapping[str, Mapping[str, list[int]]] | None = None,
        token_logprobs_by_prefix: Mapping[str, Mapping[str, list[float]]] | None = None,
    ) -> None:
        self.scores_by_prefix = dict(scores_by_prefix)
        self.token_ids_by_prefix = dict(token_ids_by_prefix or {})
        self.token_logprobs_by_prefix = dict(token_logprobs_by_prefix or {})

    def score_labels(self, prefix: str, labels: Sequence[str] = LABELS) -> ScoredLabels:
        if prefix not in self.scores_by_prefix:
            raise KeyError(f"No mock label scores were registered for prefix: {prefix!r}")

        label_scores = {
            str(label): float(self.scores_by_prefix[prefix][str(label)]) for label in labels
        }
        token_ids = self.token_ids_by_prefix.get(
            prefix,
            {str(label): [ord(str(label))] for label in labels},
        )
        token_logprobs = self.token_logprobs_by_prefix.get(
            prefix,
            {str(label): [label_scores[str(label)]] for label in labels},
        )
        return ScoredLabels(
            label_logprobs=label_scores,
            label_token_ids={str(label): list(token_ids[str(label)]) for label in labels},
            label_token_logprobs={str(label): list(token_logprobs[str(label)]) for label in labels},
        )


def evaluate_variant(
    family: Mapping[str, Any],
    variant: Mapping[str, Any],
    tokenizer: Any,
    scorer: ContinuationScorer,
    prompt_name: str,
    eps: float = TIE_EPSILON,
) -> dict[str, Any]:
    """Evaluate one logical variant with mirrored-order continuation scoring."""

    import time

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

    scores_agg = {
        "c1": 0.5 * (scores_ab["c1"] + scores_ba["c1"]),
        "c2": 0.5 * (scores_ab["c2"] + scores_ba["c2"]),
    }
    pred_winner_cid, pred_tie = predict_winner_from_scores(scores_agg, eps)

    return {
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
