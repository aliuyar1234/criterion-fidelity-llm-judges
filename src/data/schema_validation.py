from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.data.constants import (
    BIORUBRIC_ALLOWED_SLOTS,
    FAMILY_SPLITS,
    PRIMARY_CANDIDATE_IDS,
    PRIMARY_TASK_FAMILIES,
    QAKEY_RELATION_SPECS,
    TASK_FAMILY_PARAPHRASE_IDS,
    TASK_FAMILY_VARIANT_IDS,
)


class SchemaValidationError(ValueError):
    """Raised when a record violates a locked v1 schema contract."""


def _raise(path: str, message: str) -> None:
    raise SchemaValidationError(f"{path}: {message}")


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(path, "expected a mapping")
    return value


def _expect_exact_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    actual = set(value.keys())
    missing = expected - actual
    extras = actual - expected
    if missing:
        _raise(path, f"missing keys {sorted(missing)}")
    if extras:
        _raise(path, f"unexpected keys {sorted(extras)}")


def _expect_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(path, "expected a non-empty string")
    return value


def _expect_string_or_none(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _expect_non_empty_string(value, path)


def _expect_string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        _raise(path, "expected a list of strings")

    validated: list[str] = []
    for index, item in enumerate(value):
        validated.append(_expect_non_empty_string(item, f"{path}[{index}]"))
    return validated


def _expect_json_file(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                _raise(f"{path}:{line_number}", f"invalid JSON: {error.msg}")

            if not isinstance(parsed, dict):
                _raise(f"{path}:{line_number}", "expected each JSONL line to be an object")
            yield parsed


def validate_family_record(record: Mapping[str, Any]) -> None:
    """Validate the locked family-on-disk schema for v1."""

    record = _expect_mapping(record, "family")
    _expect_exact_keys(
        record,
        "family",
        {"family_id", "task_family", "split", "task_text", "candidates", "variants", "metadata"},
    )

    _expect_non_empty_string(record["family_id"], "family.family_id")
    task_family = _expect_non_empty_string(record["task_family"], "family.task_family")
    if task_family not in PRIMARY_TASK_FAMILIES:
        _raise("family.task_family", f"expected one of {sorted(PRIMARY_TASK_FAMILIES)}")

    split = _expect_non_empty_string(record["split"], "family.split")
    if split not in FAMILY_SPLITS:
        _raise("family.split", f"expected one of {sorted(FAMILY_SPLITS)}")

    _expect_non_empty_string(record["task_text"], "family.task_text")
    _expect_mapping(record["metadata"], "family.metadata")

    candidates = record["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        _raise("family.candidates", "expected exactly two candidates")

    candidate_ids: set[str] = set()
    for index, candidate_value in enumerate(candidates):
        candidate = _expect_mapping(candidate_value, f"family.candidates[{index}]")
        _expect_exact_keys(candidate, f"family.candidates[{index}]", {"cid", "text"})
        cid = _expect_non_empty_string(candidate["cid"], f"family.candidates[{index}].cid")
        _expect_non_empty_string(candidate["text"], f"family.candidates[{index}].text")
        if cid in candidate_ids:
            _raise(f"family.candidates[{index}].cid", f"duplicate candidate id {cid}")
        candidate_ids.add(cid)

    if candidate_ids != PRIMARY_CANDIDATE_IDS:
        _raise(
            "family.candidates",
            f"expected candidate ids {sorted(PRIMARY_CANDIDATE_IDS)}, got {sorted(candidate_ids)}",
        )

    variants = record["variants"]
    expected_variant_ids = set(TASK_FAMILY_VARIANT_IDS[task_family])
    paraphrase_ids = set(TASK_FAMILY_PARAPHRASE_IDS[task_family])
    if not isinstance(variants, list) or len(variants) != len(expected_variant_ids):
        _raise("family.variants", f"expected exactly {len(expected_variant_ids)} variants")

    variants_by_id: dict[str, Mapping[str, Any]] = {}
    for index, variant_value in enumerate(variants):
        variant = _expect_mapping(variant_value, f"family.variants[{index}]")
        _expect_exact_keys(
            variant,
            f"family.variants[{index}]",
            {
                "variant_id",
                "kind",
                "criterion_text",
                "semantics_id",
                "gold_winner_cid",
                "gold_scores",
                "metadata",
            },
        )

        variant_id = _expect_non_empty_string(
            variant["variant_id"], f"family.variants[{index}].variant_id"
        )
        kind = _expect_non_empty_string(variant["kind"], f"family.variants[{index}].kind")
        _expect_non_empty_string(
            variant["criterion_text"], f"family.variants[{index}].criterion_text"
        )
        _expect_non_empty_string(variant["semantics_id"], f"family.variants[{index}].semantics_id")
        gold_winner_cid = _expect_non_empty_string(
            variant["gold_winner_cid"], f"family.variants[{index}].gold_winner_cid"
        )
        _expect_mapping(variant["metadata"], f"family.variants[{index}].metadata")

        if variant_id in variants_by_id:
            _raise(f"family.variants[{index}].variant_id", f"duplicate variant id {variant_id}")
        variants_by_id[variant_id] = variant

        if variant_id == "base" and kind != "base":
            _raise(f"family.variants[{index}].kind", "base variant must have kind='base'")
        if variant_id == "counterfactual" and kind != "counterfactual":
            _raise(
                f"family.variants[{index}].kind",
                "counterfactual variant must have kind='counterfactual'",
            )
        if variant_id in paraphrase_ids and kind != "paraphrase":
            _raise(
                f"family.variants[{index}].kind",
                "paraphrase variants must have kind='paraphrase'",
            )
        if variant_id not in expected_variant_ids:
            _raise(
                f"family.variants[{index}].variant_id",
                f"unexpected variant id for task family {task_family}",
            )
        if gold_winner_cid not in candidate_ids:
            _raise(
                f"family.variants[{index}].gold_winner_cid",
                f"expected one of {sorted(candidate_ids)}",
            )

    if set(variants_by_id) != expected_variant_ids:
        _raise(
            "family.variants",
            f"expected variant ids {sorted(expected_variant_ids)}, got {sorted(variants_by_id)}",
        )

    base_variant = variants_by_id["base"]
    counterfactual_variant = variants_by_id["counterfactual"]
    base_semantics = _expect_non_empty_string(base_variant["semantics_id"], "family.variants.base")
    base_gold = _expect_non_empty_string(
        base_variant["gold_winner_cid"], "family.variants.base.gold_winner_cid"
    )

    for paraphrase_id in paraphrase_ids:
        paraphrase_variant = variants_by_id[paraphrase_id]
        paraphrase_semantics = _expect_non_empty_string(
            paraphrase_variant["semantics_id"], f"family.variants.{paraphrase_id}.semantics_id"
        )
        paraphrase_gold = _expect_non_empty_string(
            paraphrase_variant["gold_winner_cid"],
            f"family.variants.{paraphrase_id}.gold_winner_cid",
        )
        if paraphrase_semantics != base_semantics:
            _raise(
                f"family.variants.{paraphrase_id}.semantics_id",
                "paraphrase variants must preserve the base semantics_id",
            )
        if paraphrase_gold != base_gold:
            _raise(
                f"family.variants.{paraphrase_id}.gold_winner_cid",
                "paraphrase variants must preserve the base gold winner",
            )

    counterfactual_semantics = _expect_non_empty_string(
        counterfactual_variant["semantics_id"], "family.variants.counterfactual.semantics_id"
    )
    counterfactual_gold = _expect_non_empty_string(
        counterfactual_variant["gold_winner_cid"],
        "family.variants.counterfactual.gold_winner_cid",
    )
    if counterfactual_semantics == base_semantics:
        _raise(
            "family.variants.counterfactual.semantics_id",
            "counterfactual semantics_id must differ from the base semantics_id",
        )
    if counterfactual_gold == base_gold:
        _raise(
            "family.variants.counterfactual.gold_winner_cid",
            "counterfactual gold winner must differ from the base gold winner",
        )


def validate_qa_item(record: Mapping[str, Any]) -> None:
    """Validate the canonical QA item row required by the locked spec."""

    record = _expect_mapping(record, "qa_item")
    _expect_exact_keys(
        record,
        "qa_item",
        {
            "qa_id",
            "question",
            "subject_id",
            "subject_text",
            "relation_id",
            "relation_name",
            "answer_id",
            "answer_text",
            "answer_norm",
            "answer_aliases_norm",
            "coarse_type",
            "source_ref",
        },
    )

    for field_name in (
        "qa_id",
        "question",
        "subject_id",
        "subject_text",
        "relation_id",
        "relation_name",
        "answer_text",
        "answer_norm",
        "source_ref",
    ):
        _expect_non_empty_string(record[field_name], f"qa_item.{field_name}")

    _expect_string_or_none(record["answer_id"], "qa_item.answer_id")
    _expect_string_list(record["answer_aliases_norm"], "qa_item.answer_aliases_norm")

    relation_id = _expect_non_empty_string(record["relation_id"], "qa_item.relation_id")
    coarse_type = _expect_non_empty_string(record["coarse_type"], "qa_item.coarse_type")
    if coarse_type not in {"location", "date", "currency"}:
        _raise("qa_item.coarse_type", "expected one of ['currency', 'date', 'location']")

    locked_relation = QAKEY_RELATION_SPECS.get(relation_id)
    if locked_relation is not None and coarse_type != locked_relation["coarse_type"]:
        _raise(
            "qa_item.coarse_type",
            f"coarse_type must match the frozen v1 mapping for relation {relation_id}",
        )


def validate_entity_fact(record: Mapping[str, Any]) -> None:
    """Validate the canonical BioRubric fact row required by the locked spec."""

    record = _expect_mapping(record, "entity_fact")
    _expect_exact_keys(
        record,
        "entity_fact",
        {"entity_id", "entity_name", "slot", "value_id", "value_text", "value_norm", "source_ref"},
    )

    for field_name in (
        "entity_id",
        "entity_name",
        "slot",
        "value_id",
        "value_text",
        "value_norm",
        "source_ref",
    ):
        _expect_non_empty_string(record[field_name], f"entity_fact.{field_name}")

    slot = _expect_non_empty_string(record["slot"], "entity_fact.slot")
    if slot not in BIORUBRIC_ALLOWED_SLOTS:
        _raise("entity_fact.slot", f"expected one of {sorted(BIORUBRIC_ALLOWED_SLOTS)}")


def validate_jsonl_records(path: str | Path, validator: Callable[[Mapping[str, Any]], None]) -> int:
    """Validate each JSON object in a JSONL file and return the record count."""

    record_count = 0
    for record in _expect_json_file(Path(path)):
        validator(record)
        record_count += 1
    return record_count
