from __future__ import annotations

import pytest

from src.data.toy_records import TOY_QA_FAMILY
from src.inference.prompts import build_user_message
from src.inference.score_ab import (
    MockContinuationScorer,
    evaluate_variant,
    render_variant_prefix,
)


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


def _base_variant():
    return next(variant for variant in TOY_QA_FAMILY["variants"] if variant["variant_id"] == "base")


def test_prompt_user_message_ends_before_label() -> None:
    user_message = build_user_message(
        task_text="Task text",
        criterion_text="Criterion text",
        candidate_a_text="Candidate A text",
        candidate_b_text="Candidate B text",
    )

    assert user_message.endswith("Final label:")


def test_strict_prompt_prefix_contains_strict_instruction() -> None:
    tokenizer = DummyTokenizer()
    variant = _base_variant()

    prefix = render_variant_prefix(
        TOY_QA_FAMILY,
        variant,
        tokenizer,
        "strict_criterion_emphasis",
        "AB",
    )

    assert "Judge ONLY by the provided criterion." in prefix
    assert "If the criterion conflicts with what you believe, follow the criterion." in prefix


def test_mirrored_order_scoring_matches_locked_toy_numbers() -> None:
    tokenizer = DummyTokenizer()
    variant = _base_variant()
    prefix_ab = render_variant_prefix(TOY_QA_FAMILY, variant, tokenizer, "standard", "AB")
    prefix_ba = render_variant_prefix(TOY_QA_FAMILY, variant, tokenizer, "standard", "BA")
    scorer = MockContinuationScorer(
        scores_by_prefix={
            prefix_ab: {"A": -0.30, "B": -1.10},
            prefix_ba: {"A": -0.90, "B": -0.40},
        },
        token_ids_by_prefix={
            prefix_ab: {"A": [11], "B": [12]},
            prefix_ba: {"A": [11], "B": [12]},
        },
    )

    result = evaluate_variant(TOY_QA_FAMILY, variant, tokenizer, scorer, prompt_name="standard")

    assert result["scores_ab"] == {"c1": -0.30, "c2": -1.10}
    assert result["scores_ba"] == {"c1": -0.40, "c2": -0.90}
    assert result["scores_agg"] == {"c1": -0.35, "c2": -1.0}
    assert result["pred_winner_cid"] == "c1"
    assert result["order_pred_ab"] == "c1"
    assert result["order_pred_ba"] == "c1"
    assert result["label_token_ids"] == {"A": [11], "B": [12]}
    assert result["label_token_logprobs"] == {"A": [-0.30], "B": [-1.10]}


def test_exact_mirrors_aggregate_to_tie() -> None:
    tokenizer = DummyTokenizer()
    variant = _base_variant()
    prefix_ab = render_variant_prefix(TOY_QA_FAMILY, variant, tokenizer, "standard", "AB")
    prefix_ba = render_variant_prefix(TOY_QA_FAMILY, variant, tokenizer, "standard", "BA")
    scorer = MockContinuationScorer(
        scores_by_prefix={
            prefix_ab: {"A": -0.70, "B": -0.20},
            prefix_ba: {"A": -0.70, "B": -0.20},
        },
    )

    result = evaluate_variant(TOY_QA_FAMILY, variant, tokenizer, scorer, prompt_name="standard")

    assert result["scores_agg"]["c1"] == pytest.approx(-0.45)
    assert result["scores_agg"]["c2"] == pytest.approx(-0.45)
    assert result["pred_winner_cid"] == "tie"


def test_tie_threshold_returns_tie() -> None:
    tokenizer = DummyTokenizer()
    variant = _base_variant()
    prefix_ab = render_variant_prefix(TOY_QA_FAMILY, variant, tokenizer, "standard", "AB")
    prefix_ba = render_variant_prefix(TOY_QA_FAMILY, variant, tokenizer, "standard", "BA")
    scorer = MockContinuationScorer(
        scores_by_prefix={
            prefix_ab: {"A": -0.5000004, "B": -0.5000000},
            prefix_ba: {"A": -0.5000000, "B": -0.5000004},
        },
    )

    result = evaluate_variant(TOY_QA_FAMILY, variant, tokenizer, scorer, prompt_name="standard")

    assert result["pred_winner_cid"] == "tie"
    assert result["pred_tie"] is True
