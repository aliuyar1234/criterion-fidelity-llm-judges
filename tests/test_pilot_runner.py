from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.data.toy_records import TOY_BIORUBRIC_FAMILY, TOY_QA_FAMILY
from src.inference.pilot_runner import (
    build_prefix_audit_samples,
    evaluate_family_records,
    run_and_save_pilot_slice,
)
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


def _build_consistently_correct_scorer(family):
    tokenizer = DummyTokenizer()
    scores_by_prefix: dict[str, dict[str, float]] = {}
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
                label_scores = {"A": -1.0, "B": -0.1} if order == "AB" else {"A": -0.1, "B": -1.0}
            scores_by_prefix[prefix] = label_scores
    return tokenizer, MockContinuationScorer(scores_by_prefix=scores_by_prefix)


def test_evaluate_family_records_and_prefix_samples() -> None:
    tokenizer, scorer = _build_consistently_correct_scorer(TOY_QA_FAMILY)

    family_results = evaluate_family_records(
        [TOY_QA_FAMILY],
        tokenizer=tokenizer,
        scorer=scorer,
        prompt_name="standard",
    )
    samples = build_prefix_audit_samples(family_results, max_samples=2)

    assert len(family_results) == 1
    assert family_results[0]["gcf_success"] == 1
    assert len(samples) == 2
    assert samples[0]["family_id"] == TOY_QA_FAMILY["family_id"]


def test_run_and_save_pilot_slice_writes_artifacts(monkeypatch) -> None:
    tokenizer, scorer = _build_consistently_correct_scorer(TOY_QA_FAMILY)
    tmp_path = (Path(".pytest_local_tmp") / "pilot_runner").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = (tmp_path / "pilot.jsonl").resolve()
    dataset_path.write_text(json.dumps(TOY_QA_FAMILY) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = run_and_save_pilot_slice(
        task_family="qa_key",
        dataset_path=dataset_path,
        model_id="Llama-3.1-8B-Instruct",
        prompt_id="standard",
        split="dev",
        tokenizer=tokenizer,
        scorer=scorer,
        max_families=1,
    )

    assert result["n_evaluated_families"] == 1
    assert result["summary_metrics"]["gcf"] == 1.0


def test_run_and_save_pilot_slice_supports_biorubric_artifact_naming(monkeypatch) -> None:
    tokenizer, scorer = _build_consistently_correct_scorer(TOY_BIORUBRIC_FAMILY)
    tmp_path = (Path(".pytest_local_tmp") / "pilot_runner_biorubric").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = (tmp_path / "pilot.jsonl").resolve()
    dataset_path.write_text(json.dumps(TOY_BIORUBRIC_FAMILY) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = run_and_save_pilot_slice(
        task_family="biorubric",
        dataset_path=dataset_path,
        model_id="Qwen2.5-14B-Instruct",
        prompt_id="standard",
        split="test",
        tokenizer=tokenizer,
        scorer=scorer,
        max_families=1,
    )

    assert result["n_evaluated_families"] == 1
    assert result["summary_metrics"]["gcf"] == 1.0
    assert Path(result["metrics_path"]).name.startswith("biorubric__qwen2-5-14b-instruct")
