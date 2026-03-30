from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.constants import BIORUBRIC_ALLOWED_SLOTS, QAKEY_RELATION_SPECS
from src.data.schema_validation import (
    SchemaValidationError,
    validate_entity_fact,
    validate_qa_item,
)
from src.data.text_normalization import normalize_text_v1


class CanonicalSourceError(ValueError):
    """Raised when canonical-source configs or raw rows violate the locked contract."""


def normalize_text_value(value: str) -> str:
    """Apply the locked v1 normalization for answer and fact values."""

    return normalize_text_v1(value)


def compute_split_assignment(anchor_id: str) -> dict[str, int | str]:
    """Apply the frozen deterministic SHA1 split policy."""

    import hashlib

    bucket = int(hashlib.sha1(anchor_id.encode("utf-8")).hexdigest(), 16) % 20
    if bucket <= 13:
        split = "train"
    elif bucket <= 16:
        split = "dev"
    else:
        split = "test"
    return {"anchor_id": anchor_id, "bucket": bucket, "split": split}


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    file_path = Path(path)
    if not file_path.exists():
        raise CanonicalSourceError(f"Input file does not exist: {file_path}")

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise CanonicalSourceError(
                    f"{file_path}:{line_number} is not valid JSON: {error.msg}"
                ) from error
            if not isinstance(parsed, dict):
                raise CanonicalSourceError(
                    f"{file_path}:{line_number} must be a JSON object per line."
                )
            records.append(parsed)
    return records


def write_jsonl_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write dictionaries to JSONL with one object per line."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact with indentation."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def derive_sidecar_paths(output_path: str | Path) -> dict[str, Path]:
    """Derive QC and split-manifest paths next to a canonical table output."""

    output_file = Path(output_path)
    stem = output_file.stem
    return {
        "qc_report_path": output_file.with_name(f"{stem}_qc.json"),
        "split_manifest_path": output_file.with_name(f"{stem}_splits.jsonl"),
    }


def load_and_validate_relation_whitelist(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Validate the relation whitelist config against the frozen v1 mapping."""

    whitelist_entries = config.get("relation_whitelist")
    if not isinstance(whitelist_entries, list):
        raise CanonicalSourceError("QA-Key source config must contain a relation_whitelist list.")

    whitelist_map: dict[str, dict[str, str]] = {}
    for entry in whitelist_entries:
        if not isinstance(entry, dict):
            raise CanonicalSourceError("Each relation_whitelist entry must be an object.")
        relation_id = entry.get("relation_id")
        if not isinstance(relation_id, str):
            raise CanonicalSourceError(
                "Each relation_whitelist entry must have a relation_id string."
            )
        whitelist_map[relation_id] = {
            "relation_name": str(entry.get("relation_name", "")),
            "coarse_type": str(entry.get("coarse_type", "")),
            "question_template": str(entry.get("question_template", "")),
        }

    if whitelist_map != QAKEY_RELATION_SPECS:
        raise CanonicalSourceError(
            "QA-Key relation whitelist config does not match the frozen v1 mapping "
            "in METHOD_SPEC.md."
        )

    return whitelist_map


def _extract_required_string(row: Mapping[str, Any], field_name: str, *, raw_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise CanonicalSourceError(
            f"{raw_name} rows must contain a non-empty string field '{field_name}'."
        )
    return value.strip()


def _extract_optional_string(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CanonicalSourceError(f"Optional field '{field_name}' must be a string when present.")
    stripped = value.strip()
    return stripped or None


def _extract_string_list(row: Mapping[str, Any], field_name: str) -> list[str]:
    value = row.get(field_name, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise CanonicalSourceError(f"Optional field '{field_name}' must be a list of strings.")
    aliases: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise CanonicalSourceError(f"Optional field '{field_name}' must be a list of strings.")
        stripped = item.strip()
        if stripped:
            aliases.append(stripped)
    return aliases


def build_canonical_qatable(
    raw_rows: Iterable[Mapping[str, Any]],
    relation_whitelist: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the locked canonical QA table and deterministic split sidecar."""

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    qc_skip_reasons: Counter[str] = Counter()
    relation_input_counts: Counter[str] = Counter()
    duplicate_raw_row_count = 0
    kept_raw_row_count = 0
    input_row_count = 0

    for raw_row in raw_rows:
        input_row_count += 1
        subject_id = _extract_required_string(
            raw_row,
            "subject_id",
            raw_name="QA-Key raw source",
        )
        subject_text = _extract_required_string(
            raw_row,
            "subject_text",
            raw_name="QA-Key raw source",
        )
        relation_id = _extract_required_string(
            raw_row,
            "relation_id",
            raw_name="QA-Key raw source",
        )
        object_text = _extract_required_string(
            raw_row,
            "object_text",
            raw_name="QA-Key raw source",
        )
        source_ref = _extract_required_string(
            raw_row,
            "source_ref",
            raw_name="QA-Key raw source",
        )

        if relation_id not in relation_whitelist:
            qc_skip_reasons["relation_not_whitelisted"] += 1
            continue

        relation_input_counts[relation_id] += 1
        answer_norm = normalize_text_value(object_text)
        if not answer_norm:
            qc_skip_reasons["empty_normalized_answer"] += 1
            continue

        grouped_rows[(subject_id, relation_id)].append(
            {
                "subject_id": subject_id,
                "subject_text": subject_text,
                "relation_id": relation_id,
                "object_id": _extract_optional_string(raw_row, "object_id"),
                "object_text": object_text.strip(),
                "object_aliases": _extract_string_list(raw_row, "object_aliases"),
                "source_ref": source_ref,
                "answer_norm": answer_norm,
            }
        )
        kept_raw_row_count += 1

    canonical_rows: list[dict[str, Any]] = []
    split_manifest_by_anchor: dict[str, dict[str, int | str]] = {}

    for subject_id, relation_id in sorted(grouped_rows):
        relation_spec = relation_whitelist[relation_id]
        rows = grouped_rows[(subject_id, relation_id)]
        unique_answers: dict[tuple[str | None, str], dict[str, Any]] = {}

        for row in rows:
            answer_key = (row["object_id"], row["answer_norm"])
            if answer_key in unique_answers:
                duplicate_raw_row_count += 1
                continue
            unique_answers[answer_key] = row

        if len(unique_answers) != 1:
            qc_skip_reasons["multi_gold_object"] += 1
            continue

        canonical_source = next(iter(unique_answers.values()))
        answer_aliases_norm = [
            normalize_text_value(alias)
            for alias in canonical_source["object_aliases"]
            if normalize_text_value(alias)
        ]
        canonical_row = {
            "qa_id": f"{subject_id}:{relation_id}",
            "question": relation_spec["question_template"].format(
                SUBJECT=canonical_source["subject_text"]
            ),
            "subject_id": subject_id,
            "subject_text": canonical_source["subject_text"],
            "relation_id": relation_id,
            "relation_name": relation_spec["relation_name"],
            "answer_id": canonical_source["object_id"],
            "answer_text": canonical_source["object_text"],
            "answer_norm": canonical_source["answer_norm"],
            "answer_aliases_norm": sorted(set(answer_aliases_norm)),
            "coarse_type": relation_spec["coarse_type"],
            "source_ref": canonical_source["source_ref"],
        }
        try:
            validate_qa_item(canonical_row)
        except SchemaValidationError as error:
            raise CanonicalSourceError(str(error)) from error
        canonical_rows.append(canonical_row)
        split_manifest_by_anchor[subject_id] = compute_split_assignment(subject_id)

    split_manifest = [
        {
            "qa_anchor_id": anchor_id,
            "split": split_data["split"],
            "bucket": split_data["bucket"],
        }
        for anchor_id, split_data in sorted(split_manifest_by_anchor.items())
    ]

    qc_report = {
        "task_family": "qa_key",
        "input_row_count": input_row_count,
        "rows_after_whitelist_and_normalization": kept_raw_row_count,
        "group_count": len(grouped_rows),
        "output_row_count": len(canonical_rows),
        "duplicate_raw_row_count": duplicate_raw_row_count,
        "skip_reasons": dict(qc_skip_reasons),
        "relation_input_counts": dict(sorted(relation_input_counts.items())),
        "relation_output_counts": dict(
            sorted(Counter(row["relation_id"] for row in canonical_rows).items())
        ),
        "coarse_type_output_counts": dict(
            sorted(Counter(row["coarse_type"] for row in canonical_rows).items())
        ),
        "split_counts": dict(sorted(Counter(row["split"] for row in split_manifest).items())),
    }
    return canonical_rows, split_manifest, qc_report


def build_canonical_facttable(
    raw_rows: Iterable[Mapping[str, Any]],
    allowed_slots: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the locked canonical BioRubric fact table and deterministic split sidecar."""

    allowed_slot_set = set(allowed_slots)
    if allowed_slot_set != BIORUBRIC_ALLOWED_SLOTS:
        raise CanonicalSourceError(
            "BioRubric allowed_slots config does not match the frozen v1 slot set."
        )

    qc_skip_reasons: Counter[str] = Counter()
    duplicate_raw_row_count = 0
    input_row_count = 0
    canonical_rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    split_manifest_by_anchor: dict[str, dict[str, int | str]] = {}

    for raw_row in raw_rows:
        input_row_count += 1
        entity_id = _extract_required_string(
            raw_row,
            "entity_id",
            raw_name="BioRubric raw source",
        )
        entity_name = _extract_required_string(
            raw_row,
            "entity_name",
            raw_name="BioRubric raw source",
        )
        slot = _extract_required_string(
            raw_row,
            "slot",
            raw_name="BioRubric raw source",
        )
        value_id = _extract_required_string(
            raw_row,
            "value_id",
            raw_name="BioRubric raw source",
        )
        value_text = _extract_required_string(
            raw_row,
            "value_text",
            raw_name="BioRubric raw source",
        )
        source_ref = _extract_required_string(
            raw_row,
            "source_ref",
            raw_name="BioRubric raw source",
        )

        is_human = raw_row.get("is_human")
        if not isinstance(is_human, bool):
            raise CanonicalSourceError(
                "BioRubric raw source rows must include boolean field 'is_human'."
            )
        if not is_human:
            qc_skip_reasons["non_human_entity"] += 1
            continue

        if slot not in allowed_slot_set:
            qc_skip_reasons["disallowed_slot"] += 1
            continue

        value_norm = normalize_text_value(value_text)
        if not value_norm:
            qc_skip_reasons["empty_normalized_value"] += 1
            continue

        dedupe_key = (entity_id, slot, value_id)
        if dedupe_key in canonical_rows_by_key:
            duplicate_raw_row_count += 1
            continue

        canonical_row = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "slot": slot,
            "value_id": value_id,
            "value_text": value_text.strip(),
            "value_norm": value_norm,
            "source_ref": source_ref,
        }
        try:
            validate_entity_fact(canonical_row)
        except SchemaValidationError as error:
            raise CanonicalSourceError(str(error)) from error
        canonical_rows_by_key[dedupe_key] = canonical_row
        split_manifest_by_anchor[entity_id] = compute_split_assignment(entity_id)

    canonical_rows = [canonical_rows_by_key[key] for key in sorted(canonical_rows_by_key)]
    split_manifest = [
        {
            "entity_anchor_id": anchor_id,
            "split": split_data["split"],
            "bucket": split_data["bucket"],
        }
        for anchor_id, split_data in sorted(split_manifest_by_anchor.items())
    ]

    qc_report = {
        "task_family": "biorubric",
        "input_row_count": input_row_count,
        "output_row_count": len(canonical_rows),
        "duplicate_raw_row_count": duplicate_raw_row_count,
        "skip_reasons": dict(qc_skip_reasons),
        "slot_row_counts": dict(sorted(Counter(row["slot"] for row in canonical_rows).items())),
        "slot_entity_counts": {
            slot: len({row["entity_id"] for row in canonical_rows if row["slot"] == slot})
            for slot in sorted(allowed_slot_set)
        },
        "split_counts": dict(sorted(Counter(row["split"] for row in split_manifest).items())),
    }
    return canonical_rows, split_manifest, qc_report


def load_canonical_qatable(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a canonical QA table JSONL file."""

    rows = load_jsonl_records(path)
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        validate_qa_item(row)
        validated_rows.append(row)
    return validated_rows


def load_canonical_facttable(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a canonical fact table JSONL file."""

    rows = load_jsonl_records(path)
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        validate_entity_fact(row)
        validated_rows.append(row)
    return validated_rows
