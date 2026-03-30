from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.canonical_tables import (
    derive_sidecar_paths,
    load_canonical_qatable,
    load_jsonl_records,
    write_json,
    write_jsonl_records,
)
from src.data.schema_validation import SchemaValidationError, validate_family_record
from src.data.text_normalization import char_length_v1, normalize_text_v1, token_count_v1

QAKEY_NON_DATE_RELATIONS = {"P19", "P20", "P36", "P38"}
QAKEY_DATE_RELATIONS = {"P569", "P570"}
QAKEY_FORBIDDEN_CHARS = {",", ";", "/", "|", "(", ")"}
QAKEY_VARIANT_TEXTS = {
    "base": "Official answer key: {ANSWER}.",
    "para_1": "Grade using this reference answer: {ANSWER}.",
    "para_2": "Use the following answer key when judging: {ANSWER}.",
    "counterfactual": "Official answer key: {ANSWER}.",
}


class QAKeyBuildError(ValueError):
    """Raised when the locked QA-Key builder contract is violated."""


def _attach_splits(
    canonical_rows: Iterable[Mapping[str, Any]],
    split_manifest_by_anchor: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_with_split: list[dict[str, Any]] = []
    for row in canonical_rows:
        subject_id = str(row["subject_id"])
        split_data = split_manifest_by_anchor.get(subject_id)
        if split_data is None:
            raise QAKeyBuildError(
                f"Canonical QA row {row['qa_id']!r} has no matching split manifest entry "
                f"for subject_id {subject_id!r}."
            )

        enriched = dict(row)
        enriched["split"] = split_data["split"]
        enriched["split_bucket"] = split_data["bucket"]
        rows_with_split.append(enriched)
    return rows_with_split


def _contains_forbidden_qakey_chars(value: str) -> bool:
    return any(character in value for character in QAKEY_FORBIDDEN_CHARS)


def _build_donor_check_map(
    anchor_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
) -> dict[str, Any]:
    donor_norm = str(donor_row["answer_norm"])
    gold_norm = str(anchor_row["answer_norm"])
    relation_id = str(anchor_row["relation_id"])

    checks = {
        "same_relation_id": donor_row["relation_id"] == anchor_row["relation_id"],
        "same_coarse_type": donor_row["coarse_type"] == anchor_row["coarse_type"],
        "same_split": donor_row["split"] == anchor_row["split"],
        "different_subject_id": donor_row["subject_id"] != anchor_row["subject_id"],
        "different_answer_id": (
            anchor_row["answer_id"] is None
            or donor_row["answer_id"] is None
            or donor_row["answer_id"] != anchor_row["answer_id"]
        ),
        "different_answer_norm": donor_norm != gold_norm,
        "different_from_gold_aliases": donor_norm not in set(anchor_row["answer_aliases_norm"]),
        "forbidden_char_absent": not _contains_forbidden_qakey_chars(donor_norm),
        "forbidden_and_absent": " and " not in donor_norm,
    }

    donor_token_count = token_count_v1(str(donor_row["answer_text"]))
    donor_char_len = char_length_v1(str(donor_row["answer_text"]))
    gold_char_len = char_length_v1(str(anchor_row["answer_text"]))
    char_length_ratio = donor_char_len / gold_char_len if gold_char_len > 0 else float("inf")

    checks["donor_token_count"] = donor_token_count
    checks["donor_char_len"] = donor_char_len
    checks["gold_char_len"] = gold_char_len
    checks["char_length_ratio"] = char_length_ratio

    if relation_id in QAKEY_NON_DATE_RELATIONS:
        checks["token_count_in_range"] = 1 <= donor_token_count <= 6
        checks["char_length_in_range"] = 2 <= donor_char_len <= 40
    elif relation_id in QAKEY_DATE_RELATIONS:
        checks["token_count_in_range"] = 1 <= donor_token_count <= 4
        checks["char_length_in_range"] = 4 <= donor_char_len <= 20
    else:  # pragma: no cover - canonical QA table should already enforce this
        raise QAKeyBuildError(f"Unexpected relation_id in QA-Key builder: {relation_id!r}")

    checks["char_length_ratio_in_range"] = 0.67 <= char_length_ratio <= 1.50
    checks["is_admissible"] = all(
        bool(checks[key])
        for key in (
            "same_relation_id",
            "same_coarse_type",
            "same_split",
            "different_subject_id",
            "different_answer_id",
            "different_answer_norm",
            "different_from_gold_aliases",
            "token_count_in_range",
            "char_length_in_range",
            "forbidden_char_absent",
            "forbidden_and_absent",
            "char_length_ratio_in_range",
        )
    )
    return checks


def _donor_ranking_key(
    anchor_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
) -> tuple[Any, ...]:
    donor_identifier = donor_row["answer_id"] or donor_row["qa_id"]
    return (
        abs(
            token_count_v1(str(donor_row["answer_text"]))
            - token_count_v1(str(anchor_row["answer_text"]))
        ),
        abs(
            char_length_v1(str(donor_row["answer_text"]))
            - char_length_v1(str(anchor_row["answer_text"]))
        ),
        normalize_text_v1(str(donor_row["answer_text"])),
        str(donor_identifier),
    )


def select_qakey_donor(
    anchor_row: Mapping[str, Any],
    donor_pool: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the deterministic donor choice plus per-donor admissibility checks."""

    donor_evaluations: list[dict[str, Any]] = []
    admissible_donors: list[Mapping[str, Any]] = []

    for donor_row in donor_pool:
        checks = _build_donor_check_map(anchor_row, donor_row)
        donor_evaluations.append(
            {
                "donor_qa_id": donor_row["qa_id"],
                "donor_subject_id": donor_row["subject_id"],
                "donor_answer_id": donor_row["answer_id"],
                "donor_answer_text": donor_row["answer_text"],
                "donor_answer_norm": donor_row["answer_norm"],
                "checks": checks,
            }
        )
        if checks["is_admissible"]:
            admissible_donors.append(donor_row)

    if not admissible_donors:
        return None, donor_evaluations

    chosen_donor = min(
        admissible_donors,
        key=lambda donor: _donor_ranking_key(anchor_row, donor),
    )
    return dict(chosen_donor), donor_evaluations


def _render_candidate_text(answer_text: str) -> str:
    return f"The answer is {answer_text}."


def build_qakey_family(
    anchor_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
    donor_evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct one locked QA-Key family record."""

    family = {
        "family_id": f"qakey_{str(anchor_row['qa_id']).replace(':', '_')}",
        "task_family": "qa_key",
        "split": anchor_row["split"],
        "task_text": anchor_row["question"],
        "candidates": [
            {"cid": "c1", "text": _render_candidate_text(str(anchor_row["answer_text"]))},
            {"cid": "c2", "text": _render_candidate_text(str(donor_row["answer_text"]))},
        ],
        "variants": [
            {
                "variant_id": "base",
                "kind": "base",
                "criterion_text": QAKEY_VARIANT_TEXTS["base"].format(
                    ANSWER=anchor_row["answer_text"]
                ),
                "semantics_id": "s_base",
                "gold_winner_cid": "c1",
                "gold_scores": None,
                "metadata": {"template_id": "qa_base"},
            },
            {
                "variant_id": "para_1",
                "kind": "paraphrase",
                "criterion_text": QAKEY_VARIANT_TEXTS["para_1"].format(
                    ANSWER=anchor_row["answer_text"]
                ),
                "semantics_id": "s_base",
                "gold_winner_cid": "c1",
                "gold_scores": None,
                "metadata": {"template_id": "qa_para_1"},
            },
            {
                "variant_id": "para_2",
                "kind": "paraphrase",
                "criterion_text": QAKEY_VARIANT_TEXTS["para_2"].format(
                    ANSWER=anchor_row["answer_text"]
                ),
                "semantics_id": "s_base",
                "gold_winner_cid": "c1",
                "gold_scores": None,
                "metadata": {"template_id": "qa_para_2"},
            },
            {
                "variant_id": "counterfactual",
                "kind": "counterfactual",
                "criterion_text": QAKEY_VARIANT_TEXTS["counterfactual"].format(
                    ANSWER=donor_row["answer_text"]
                ),
                "semantics_id": "s_counterfactual",
                "gold_winner_cid": "c2",
                "gold_scores": None,
                "metadata": {"template_id": "qa_counterfactual"},
            },
        ],
        "metadata": {
            "qc_passed": True,
            "source_row_ids": [anchor_row["qa_id"], donor_row["qa_id"]],
            "source_refs": [anchor_row["source_ref"], donor_row["source_ref"]],
            "relation_id": anchor_row["relation_id"],
            "relation_name": anchor_row["relation_name"],
            "coarse_type": anchor_row["coarse_type"],
            "anchor_subject_id": anchor_row["subject_id"],
            "anchor_answer_id": anchor_row["answer_id"],
            "anchor_answer_norm": anchor_row["answer_norm"],
            "anchor_answer_aliases_norm": list(anchor_row["answer_aliases_norm"]),
            "donor_subject_id": donor_row["subject_id"],
            "donor_answer_id": donor_row["answer_id"],
            "donor_answer_norm": donor_row["answer_norm"],
            "donor_split": donor_row["split"],
            "selected_donor_ranking_key": list(_donor_ranking_key(anchor_row, donor_row)),
            "selected_donor_checks": next(
                evaluation["checks"]
                for evaluation in donor_evaluations
                if evaluation["donor_qa_id"] == donor_row["qa_id"]
            ),
            "donor_evaluations": donor_evaluations,
        },
    }

    try:
        validate_family_record(family)
    except SchemaValidationError as error:
        raise QAKeyBuildError(str(error)) from error

    return family


def _make_qakey_invalid_record(
    anchor_row: Mapping[str, Any],
    donor_evaluations: list[dict[str, Any]],
    *,
    skip_reason: str,
) -> dict[str, Any]:
    donor_failure_counts: Counter[str] = Counter()
    for evaluation in donor_evaluations:
        for check_name, check_value in evaluation["checks"].items():
            if check_name == "is_admissible":
                continue
            if isinstance(check_value, bool) and not check_value:
                donor_failure_counts[check_name] += 1

    return {
        "task_family": "qa_key",
        "family_id": f"qakey_{str(anchor_row['qa_id']).replace(':', '_')}",
        "qa_id": anchor_row["qa_id"],
        "subject_id": anchor_row["subject_id"],
        "subject_text": anchor_row["subject_text"],
        "split": anchor_row["split"],
        "relation_id": anchor_row["relation_id"],
        "coarse_type": anchor_row["coarse_type"],
        "skip_reason": skip_reason,
        "source_ref": anchor_row["source_ref"],
        "answer_text": anchor_row["answer_text"],
        "answer_norm": anchor_row["answer_norm"],
        "donor_pool_size": len(donor_evaluations),
        "donor_failure_counts": dict(sorted(donor_failure_counts.items())),
    }


def build_qakey_families_detailed(
    canonical_rows: Iterable[Mapping[str, Any]],
    split_manifest_rows: Iterable[Mapping[str, Any]],
    *,
    target_family_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build deterministic QA-Key families plus a QC summary."""

    split_manifest_by_anchor = _load_qakey_split_manifest_rows(split_manifest_rows)
    rows_with_split = _attach_splits(canonical_rows, split_manifest_by_anchor)
    rows_with_split = sorted(rows_with_split, key=lambda row: str(row["qa_id"]))

    rows_by_pool_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_with_split:
        pool_key = (str(row["relation_id"]), str(row["coarse_type"]), str(row["split"]))
        rows_by_pool_key[pool_key].append(row)

    skip_reasons: Counter[str] = Counter()
    donor_failure_counts: Counter[str] = Counter()
    built_families: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for anchor_row in rows_with_split:
        donor_pool_key = (
            str(anchor_row["relation_id"]),
            str(anchor_row["coarse_type"]),
            str(anchor_row["split"]),
        )
        donor_pool = rows_by_pool_key[donor_pool_key]
        donor_row, donor_evaluations = select_qakey_donor(anchor_row, donor_pool)

        if donor_row is None:
            skip_reasons["no_admissible_donor"] += 1
            for evaluation in donor_evaluations:
                for check_name, check_value in evaluation["checks"].items():
                    if check_name == "is_admissible":
                        continue
                    if isinstance(check_value, bool) and not check_value:
                        donor_failure_counts[check_name] += 1
            invalid_records.append(
                _make_qakey_invalid_record(
                    anchor_row,
                    donor_evaluations,
                    skip_reason="no_admissible_donor",
                )
            )
            continue

        built_families.append(build_qakey_family(anchor_row, donor_row, donor_evaluations))

    selected_families = built_families
    if target_family_count is not None:
        selected_families = built_families[:target_family_count]
        if len(built_families) < target_family_count:
            skip_reasons["target_family_count_not_reached"] += target_family_count - len(
                built_families
            )

    qc_report = {
        "task_family": "qa_key",
        "input_row_count": len(rows_with_split),
        "split_manifest_count": len(split_manifest_by_anchor),
        "built_family_count": len(built_families),
        "selected_family_count": len(selected_families),
        "target_family_count": target_family_count,
        "skip_reasons": dict(skip_reasons),
        "donor_failure_counts": dict(donor_failure_counts),
        "selected_split_counts": dict(
            sorted(Counter(family["split"] for family in selected_families).items())
        ),
        "selected_relation_counts": dict(
            sorted(
                Counter(family["metadata"]["relation_id"] for family in selected_families).items()
            )
        ),
        "selection_policy": "sorted_by_family_id_then_truncate",
    }

    return selected_families, qc_report, invalid_records


def build_qakey_families(
    canonical_rows: Iterable[Mapping[str, Any]],
    split_manifest_rows: Iterable[Mapping[str, Any]],
    *,
    target_family_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families, qc_report, _ = build_qakey_families_detailed(
        canonical_rows,
        split_manifest_rows,
        target_family_count=target_family_count,
    )
    return families, qc_report


def _load_qakey_split_manifest_rows(
    split_manifest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_anchor_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(split_manifest_rows):
        qa_anchor_id = record.get("qa_anchor_id")
        split = record.get("split")
        bucket = record.get("bucket")
        if not isinstance(qa_anchor_id, str) or not qa_anchor_id:
            raise QAKeyBuildError(
                f"Split manifest row {index} must contain non-empty string 'qa_anchor_id'."
            )
        if split not in {"train", "dev", "test"}:
            raise QAKeyBuildError(
                f"Split manifest row {index} has invalid split {split!r}; "
                "expected one of train/dev/test."
            )
        if not isinstance(bucket, int):
            raise QAKeyBuildError(f"Split manifest row {index} must contain integer 'bucket'.")
        if qa_anchor_id in by_anchor_id:
            raise QAKeyBuildError(
                f"Split manifest contains duplicate QA anchor id {qa_anchor_id!r}."
            )
        by_anchor_id[qa_anchor_id] = {
            "qa_anchor_id": qa_anchor_id,
            "split": split,
            "bucket": bucket,
        }
    return by_anchor_id


def build_qakey_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the locked QA-Key builder from config and write its artifacts."""

    input_path = config.get("input_path")
    output_path = config.get("output_path")
    target_family_count = config.get("target_family_count")
    if not isinstance(input_path, str) or not input_path:
        raise QAKeyBuildError("QA-Key pilot config must contain a non-empty string input_path.")
    if not isinstance(output_path, str) or not output_path:
        raise QAKeyBuildError("QA-Key pilot config must contain a non-empty string output_path.")
    if target_family_count is not None and (
        not isinstance(target_family_count, int) or target_family_count <= 0
    ):
        raise QAKeyBuildError("target_family_count must be a positive integer when present.")

    split_manifest_path = config.get("split_manifest_path")
    if split_manifest_path is None:
        split_manifest_path = str(derive_sidecar_paths(input_path)["split_manifest_path"])
    if not isinstance(split_manifest_path, str) or not split_manifest_path:
        raise QAKeyBuildError("split_manifest_path must be a non-empty string when present.")

    canonical_rows = load_canonical_qatable(input_path)
    split_manifest_rows = load_jsonl_records(split_manifest_path)
    families, qc_report, _ = build_qakey_families_detailed(
        canonical_rows,
        split_manifest_rows,
        target_family_count=target_family_count,
    )

    write_jsonl_records(output_path, families)
    qc_report_path = Path(output_path).with_name(f"{Path(output_path).stem}_qc.json")
    write_json(qc_report_path, qc_report)

    return {
        "output_path": str(output_path),
        "qc_report_path": str(qc_report_path),
        "family_count": len(families),
        "target_family_count": target_family_count,
        "split_manifest_path": split_manifest_path,
        "qc_report": qc_report,
    }
