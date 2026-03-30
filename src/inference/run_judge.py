from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from src.common.artifacts import token_id_audit_path
from src.data.constants import TIE_EPSILON
from src.data.toy_records import TOY_QA_FAMILY
from src.eval.metrics import compute_family_result
from src.inference.score_ab import (
    PRIMARY_MODEL_REPO_IDS,
    ContinuationScorer,
    evaluate_variant,
    extract_continuation_token_ids,
)


def evaluate_family(
    family: Mapping[str, Any],
    tokenizer: Any,
    scorer: ContinuationScorer,
    prompt_name: str,
    eps: float = TIE_EPSILON,
) -> dict[str, Any]:
    """Evaluate every logical variant in one family and return the family-level result."""

    variant_results = [
        evaluate_variant(
            family=family,
            variant=variant,
            tokenizer=tokenizer,
            scorer=scorer,
            prompt_name=prompt_name,
            eps=eps,
        )
        for variant in family["variants"]
    ]
    return compute_family_result(family, variant_results)


def build_token_id_audit_record(
    model_id: str,
    tokenizer: Any,
    prompt_name: str = "standard",
) -> dict[str, Any]:
    """Create one toy token-ID audit record for the base QA-Key prompt."""

    base_variant = next(
        variant for variant in TOY_QA_FAMILY["variants"] if variant["variant_id"] == "base"
    )
    from src.inference.score_ab import render_variant_prefix

    prefix = render_variant_prefix(
        family=TOY_QA_FAMILY,
        variant=base_variant,
        tokenizer=tokenizer,
        prompt_name=prompt_name,
        order="AB",
    )

    label_token_ids = {
        label: extract_continuation_token_ids(tokenizer, prefix=prefix, label=label)[2]
        for label in ("A", "B")
    }

    return {
        "model_id": model_id,
        "model_repo_id": PRIMARY_MODEL_REPO_IDS[model_id],
        "prompt_name": prompt_name,
        "family_id": TOY_QA_FAMILY["family_id"],
        "variant_id": base_variant["variant_id"],
        "order": "AB",
        "rendered_prefix": prefix,
        "label_token_ids": label_token_ids,
    }


def write_token_id_audits(
    output_path: str | Path | None = None,
    prompt_name: str = "standard",
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load the primary-model tokenizers and write one toy audit record per model."""

    audit_records = []
    for model_id, repo_id in PRIMARY_MODEL_REPO_IDS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            local_files_only=local_files_only,
            use_fast=True,
        )
        audit_records.append(
            build_token_id_audit_record(
                model_id=model_id,
                tokenizer=tokenizer,
                prompt_name=prompt_name,
            )
        )

    destination = (
        Path(output_path)
        if output_path is not None
        else token_id_audit_path(
            model_id="primary_models",
            prompt_id=prompt_name,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit_records, indent=2), encoding="utf-8")
    return audit_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run toy token audits and evaluation helpers.")
    parser.add_argument(
        "--token_audit",
        action="store_true",
        help="Write a toy token-ID audit artifact for the primary models.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the token-ID audit artifact.",
    )
    parser.add_argument(
        "--prompt_name",
        default="standard",
        choices=["standard", "strict_criterion_emphasis"],
        help="Prompt template to use for the audit prompt.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require tokenizers to be available in the local Hugging Face cache.",
    )
    args = parser.parse_args()

    if not args.token_audit:
        parser.error("No action specified. Use --token_audit.")

    audit_records = write_token_id_audits(
        output_path=args.output,
        prompt_name=args.prompt_name,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(audit_records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
