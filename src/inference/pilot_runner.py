from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.common.artifacts import metrics_path, raw_output_path
from src.data.canonical_tables import load_jsonl_records
from src.data.schema_validation import validate_family_record
from src.eval.metrics import compute_summary_metrics
from src.inference.run_judge import evaluate_family


class PilotRunError(ValueError):
    """Raised when a pilot inference run cannot be completed honestly."""


def load_family_records(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    """Load and validate family JSONL records, optionally filtering by split."""

    families = load_jsonl_records(path)
    validated: list[dict[str, Any]] = []
    for family in families:
        validate_family_record(family)
        if split is None or family["split"] == split:
            validated.append(dict(family))

    if not validated:
        raise PilotRunError("No family records were available for the requested dataset/split.")
    return validated


def evaluate_family_records(
    families: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    scorer: Any,
    prompt_name: str,
) -> list[dict[str, Any]]:
    """Run family-level evaluation for one model/prompt slice."""

    family_results = [
        evaluate_family(
            family=family,
            tokenizer=tokenizer,
            scorer=scorer,
            prompt_name=prompt_name,
        )
        for family in families
    ]
    return family_results


def build_prefix_audit_samples(
    family_results: Iterable[Mapping[str, Any]],
    *,
    max_samples: int = 5,
) -> list[dict[str, Any]]:
    """Collect a small rendered-prefix audit sample from family results."""

    samples: list[dict[str, Any]] = []
    for family_result in family_results:
        for variant_result in family_result["variant_results"]:
            samples.append(
                {
                    "family_id": family_result["family_id"],
                    "variant_id": variant_result["variant_id"],
                    "rendered_prefix_ab": variant_result["rendered_prefix_ab"],
                    "rendered_prefix_ba": variant_result["rendered_prefix_ba"],
                }
            )
            if len(samples) >= max_samples:
                return samples
    return samples


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def run_and_save_pilot_slice(
    *,
    task_family: str,
    dataset_path: str | Path,
    model_id: str,
    prompt_id: str,
    split: str,
    tokenizer: Any,
    scorer: Any,
    max_families: int | None = None,
) -> dict[str, Any]:
    """Evaluate one pilot slice and persist raw outputs, metrics, and audit samples."""

    families = load_family_records(dataset_path, split=split)
    if max_families is not None:
        families = families[:max_families]
    if not families:
        raise PilotRunError("No family records remain after applying max_families.")

    family_results = evaluate_family_records(
        families,
        tokenizer=tokenizer,
        scorer=scorer,
        prompt_name=prompt_id,
    )
    summary = compute_summary_metrics(family_results)
    audit_samples = build_prefix_audit_samples(family_results)

    raw_path = raw_output_path(task_family, model_id, prompt_id, split)
    metrics_output_path = metrics_path(task_family, model_id, prompt_id, split)
    prefix_audit_path = raw_path.with_name(f"{raw_path.stem}__prefix_audit.json")

    write_jsonl(raw_path, family_results)
    write_json(
        metrics_output_path,
        {
            "task_family": task_family,
            "model_id": model_id,
            "prompt_id": prompt_id,
            "split": split,
            "dataset_path": str(dataset_path),
            "n_evaluated_families": len(family_results),
            "summary_metrics": summary,
        },
    )
    write_json(prefix_audit_path, {"samples": audit_samples})

    return {
        "raw_output_path": str(raw_path),
        "metrics_path": str(metrics_output_path),
        "prefix_audit_path": str(prefix_audit_path),
        "n_evaluated_families": len(family_results),
        "summary_metrics": summary,
    }
