from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.canonical_tables import (
    derive_sidecar_paths,
    load_canonical_facttable,
    load_jsonl_records,
    write_json,
    write_jsonl_records,
)
from src.data.constants import BIORUBRIC_DISTINGUISHING_SLOT_PRIORITY, BIORUBRIC_SHARED_SLOT
from src.data.schema_validation import SchemaValidationError, validate_family_record
from src.data.text_normalization import char_length_v1, normalize_text_v1, token_count_v1

BIORUBRIC_FORBIDDEN_CHARS = {",", ";", "/", "|", "(", ")"}
BIORUBRIC_CANDIDATE_TEMPLATE_IDS = {
    "occupation": "occupation_sentence_v1",
    "notable_work": "notable_work_sentence_v1",
    "award": "award_sentence_v1",
    "field": "field_sentence_v1",
}
BIORUBRIC_RUBRIC_TEMPLATE_IDS = {0: "rubric_wording_set_0", 1: "rubric_wording_set_1"}


class BioRubricBuildError(ValueError):
    """Raised when the locked BioRubric builder contract is violated."""


def _load_biorubric_split_manifest_rows(
    split_manifest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_anchor_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(split_manifest_rows):
        entity_anchor_id = record.get("entity_anchor_id")
        split = record.get("split")
        bucket = record.get("bucket")
        if not isinstance(entity_anchor_id, str) or not entity_anchor_id:
            raise BioRubricBuildError(
                f"Split manifest row {index} must contain non-empty string 'entity_anchor_id'."
            )
        if split not in {"train", "dev", "test"}:
            raise BioRubricBuildError(
                f"Split manifest row {index} has invalid split {split!r}; "
                "expected one of train/dev/test."
            )
        if not isinstance(bucket, int):
            raise BioRubricBuildError(f"Split manifest row {index} must contain integer 'bucket'.")
        if entity_anchor_id in by_anchor_id:
            raise BioRubricBuildError(
                f"Split manifest contains duplicate entity anchor id {entity_anchor_id!r}."
            )
        by_anchor_id[entity_anchor_id] = {
            "entity_anchor_id": entity_anchor_id,
            "split": split,
            "bucket": bucket,
        }
    return by_anchor_id


def _attach_splits(
    canonical_rows: Iterable[Mapping[str, Any]],
    split_manifest_by_anchor: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_with_split: list[dict[str, Any]] = []
    for row in canonical_rows:
        entity_id = str(row["entity_id"])
        split_data = split_manifest_by_anchor.get(entity_id)
        if split_data is None:
            raise BioRubricBuildError(
                f"Canonical fact row for entity_id {entity_id!r} has no matching split manifest."
            )
        enriched = dict(row)
        enriched["split"] = split_data["split"]
        enriched["split_bucket"] = split_data["bucket"]
        rows_with_split.append(enriched)
    return rows_with_split


def _is_admissible_value_text(value_text: str) -> bool:
    normalized = normalize_text_v1(value_text)
    if not normalized:
        return False
    if not (1 <= token_count_v1(value_text) <= 8):
        return False
    if not (2 <= char_length_v1(value_text) <= 60):
        return False
    if any(character in normalized for character in BIORUBRIC_FORBIDDEN_CHARS):
        return False
    if " and " in normalized:
        return False
    return True


def _row_identifier(row: Mapping[str, Any]) -> str:
    return f"{row['entity_id']}:{row['slot']}:{row['value_id']}"


def _canonicalize_fact_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_norm: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = str(row["value_norm"])
        candidate = {
            "entity_id": row["entity_id"],
            "entity_name": row["entity_name"],
            "slot": row["slot"],
            "value_id": row["value_id"],
            "value_text": row["value_text"],
            "value_norm": row["value_norm"],
            "source_ref": row["source_ref"],
            "split": row["split"],
            "split_bucket": row["split_bucket"],
        }
        current = by_norm.get(normalized)
        if current is None or (
            str(candidate["value_text"]),
            str(candidate["value_id"]),
            str(candidate["source_ref"]),
        ) < (
            str(current["value_text"]),
            str(current["value_id"]),
            str(current["source_ref"]),
        ):
            by_norm[normalized] = candidate
    return [by_norm[key] for key in sorted(by_norm)]


def _render_candidate_sentence(slot: str, *, entity_name: str, value_text: str) -> str:
    if slot == "occupation":
        return f"{entity_name} was a {value_text}."
    if slot == "notable_work":
        return f"{entity_name} is known for {value_text}."
    if slot == "award":
        return f"{entity_name} received {value_text}."
    if slot == "field":
        return f"{entity_name} worked in {value_text}."
    raise BioRubricBuildError(f"Unexpected slot for candidate rendering: {slot!r}")


def _render_rubric_item(
    *,
    wording_set: int,
    kind: str,
    weight: int,
    entity_name: str,
    value_text: str,
) -> str:
    if wording_set == 0:
        if kind == "shared":
            return (
                f"Award {weight} point(s) if the biography states that {entity_name} "
                f"was a {value_text}."
            )
        if kind in {"f1", "f2"}:
            return f"Award {weight} point(s) if the biography mentions {value_text}."
    if wording_set == 1:
        if kind == "shared":
            return f"Give {weight} point(s) for stating that {entity_name} was a {value_text}."
        if kind in {"f1", "f2"}:
            return f"Give {weight} point(s) for referencing {value_text}."
    raise BioRubricBuildError(
        f"Unexpected rubric item request wording_set={wording_set!r}, kind={kind!r}."
    )


def _render_rubric_text(
    *,
    wording_set: int,
    order: Iterable[str],
    weights: Mapping[str, int],
    entity_name: str,
    shared_value_text: str,
    f1_value_text: str,
    f2_value_text: str,
) -> str:
    value_lookup = {"shared": shared_value_text, "f1": f1_value_text, "f2": f2_value_text}
    lines = [
        _render_rubric_item(
            wording_set=wording_set,
            kind=item_id,
            weight=int(weights[item_id]),
            entity_name=entity_name,
            value_text=value_lookup[item_id],
        )
        for item_id in order
    ]
    return "\n".join(lines)


def _build_candidate_text(
    *,
    entity_name: str,
    shared_value_text: str,
    slot: str,
    value_text: str,
) -> str:
    shared_sentence = _render_candidate_sentence(
        "occupation",
        entity_name=entity_name,
        value_text=shared_value_text,
    )
    distinguishing_sentence = _render_candidate_sentence(
        slot,
        entity_name=entity_name,
        value_text=value_text,
    )
    return f"{shared_sentence} {distinguishing_sentence}"


def _candidate_length_ratio(candidate_a_text: str, candidate_b_text: str) -> float:
    a_count = token_count_v1(candidate_a_text)
    b_count = token_count_v1(candidate_b_text)
    shorter = min(a_count, b_count)
    longer = max(a_count, b_count)
    if shorter <= 0:
        return float("inf")
    return longer / shorter


def _is_exactly_two_template_sentences(
    *,
    candidate_text: str,
    shared_sentence: str,
    distinguishing_sentence: str,
) -> bool:
    return (
        shared_sentence.endswith(".")
        and distinguishing_sentence.endswith(".")
        and candidate_text == f"{shared_sentence} {distinguishing_sentence}"
    )


def _build_entity_family(entity_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [dict(row) for row in entity_rows]
    if not rows:
        return None

    entity_name = str(rows[0]["entity_name"])
    entity_id = str(rows[0]["entity_id"])
    split = str(rows[0]["split"])

    rows_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_slot[str(row["slot"])].append(row)

    shared_rows = _canonicalize_fact_rows(
        row
        for row in rows_by_slot.get(BIORUBRIC_SHARED_SLOT, [])
        if _is_admissible_value_text(str(row["value_text"]))
    )
    if not shared_rows:
        return None
    shared_row = shared_rows[0]

    chosen_slot: str | None = None
    chosen_distinguishing_rows: list[dict[str, Any]] | None = None
    for slot in BIORUBRIC_DISTINGUISHING_SLOT_PRIORITY:
        admissible_rows = _canonicalize_fact_rows(
            row
            for row in rows_by_slot.get(slot, [])
            if _is_admissible_value_text(str(row["value_text"]))
        )
        if len(admissible_rows) >= 2:
            chosen_slot = slot
            chosen_distinguishing_rows = admissible_rows[:2]
            break

    if chosen_slot is None or chosen_distinguishing_rows is None:
        return None

    f1_row, f2_row = chosen_distinguishing_rows
    candidate_c1_text = _build_candidate_text(
        entity_name=entity_name,
        shared_value_text=str(shared_row["value_text"]),
        slot=chosen_slot,
        value_text=str(f1_row["value_text"]),
    )
    candidate_c2_text = _build_candidate_text(
        entity_name=entity_name,
        shared_value_text=str(shared_row["value_text"]),
        slot=chosen_slot,
        value_text=str(f2_row["value_text"]),
    )

    length_ratio = _candidate_length_ratio(candidate_c1_text, candidate_c2_text)
    if length_ratio > 1.25:
        return None

    weight_vectors = {
        "base": {"shared": 1, "f1": 2, "f2": 1},
        "para_lex": {"shared": 1, "f1": 2, "f2": 1},
        "para_struct": {"shared": 1, "f1": 2, "f2": 1},
        "counterfactual": {"shared": 1, "f1": 1, "f2": 2},
    }
    rubric_orders = {
        "base": ["shared", "f1", "f2"],
        "para_lex": ["shared", "f1", "f2"],
        "para_struct": ["f1", "shared", "f2"],
        "counterfactual": ["shared", "f1", "f2"],
    }
    wording_sets = {"base": 0, "para_lex": 1, "para_struct": 0, "counterfactual": 0}

    shared_sentence = _render_candidate_sentence(
        "occupation",
        entity_name=entity_name,
        value_text=str(shared_row["value_text"]),
    )
    distinguishing_sentence_c1 = _render_candidate_sentence(
        chosen_slot,
        entity_name=entity_name,
        value_text=str(f1_row["value_text"]),
    )
    distinguishing_sentence_c2 = _render_candidate_sentence(
        chosen_slot,
        entity_name=entity_name,
        value_text=str(f2_row["value_text"]),
    )
    distinguishing_template_id = BIORUBRIC_CANDIDATE_TEMPLATE_IDS[chosen_slot]

    family = {
        "family_id": f"biorubric_{entity_id}",
        "task_family": "biorubric",
        "split": split,
        "task_text": f"Write a two-sentence biography of {entity_name}.",
        "candidates": [
            {"cid": "c1", "text": candidate_c1_text},
            {"cid": "c2", "text": candidate_c2_text},
        ],
        "variants": [
            {
                "variant_id": variant_id,
                "kind": (
                    "base"
                    if variant_id == "base"
                    else "counterfactual"
                    if variant_id == "counterfactual"
                    else "paraphrase"
                ),
                "criterion_text": _render_rubric_text(
                    wording_set=wording_sets[variant_id],
                    order=rubric_orders[variant_id],
                    weights=weight_vectors[variant_id],
                    entity_name=entity_name,
                    shared_value_text=str(shared_row["value_text"]),
                    f1_value_text=str(f1_row["value_text"]),
                    f2_value_text=str(f2_row["value_text"]),
                ),
                "semantics_id": "s_base" if variant_id != "counterfactual" else "s_counterfactual",
                "gold_winner_cid": "c1" if variant_id != "counterfactual" else "c2",
                "gold_scores": dict(weight_vectors[variant_id]),
                "metadata": {
                    "wording_set": wording_sets[variant_id],
                    "wording_template_id": BIORUBRIC_RUBRIC_TEMPLATE_IDS[wording_sets[variant_id]],
                    "rubric_order": list(rubric_orders[variant_id]),
                    "weight_vector": dict(weight_vectors[variant_id]),
                },
            }
            for variant_id in ("base", "para_lex", "para_struct", "counterfactual")
        ],
        "metadata": {
            "qc_passed": True,
            "source_row_ids": [
                _row_identifier(shared_row),
                _row_identifier(f1_row),
                _row_identifier(f2_row),
            ],
            "source_refs": [
                str(shared_row["source_ref"]),
                str(f1_row["source_ref"]),
                str(f2_row["source_ref"]),
            ],
            "entity_id": entity_id,
            "entity_name": entity_name,
            "shared_slot": BIORUBRIC_SHARED_SLOT,
            "distinguishing_slot": chosen_slot,
            "shared_fact": {
                "value_id": shared_row["value_id"],
                "value_text": shared_row["value_text"],
                "value_norm": shared_row["value_norm"],
                "template_id": BIORUBRIC_CANDIDATE_TEMPLATE_IDS["occupation"],
            },
            "distinguishing_facts": {
                "c1": {
                    "value_id": f1_row["value_id"],
                    "value_text": f1_row["value_text"],
                    "value_norm": f1_row["value_norm"],
                    "template_id": distinguishing_template_id,
                },
                "c2": {
                    "value_id": f2_row["value_id"],
                    "value_text": f2_row["value_text"],
                    "value_norm": f2_row["value_norm"],
                    "template_id": distinguishing_template_id,
                },
            },
            "coverage_vectors": {
                "c1": {"shared": 1, "f1": 1, "f2": 0},
                "c2": {"shared": 1, "f1": 0, "f2": 1},
            },
            "variant_weight_vectors": {key: dict(value) for key, value in weight_vectors.items()},
            "candidate_checks": {
                "shared_sentence_identical": True,
                "same_distinguishing_template": True,
                "c1_exactly_two_sentences": _is_exactly_two_template_sentences(
                    candidate_text=candidate_c1_text,
                    shared_sentence=shared_sentence,
                    distinguishing_sentence=distinguishing_sentence_c1,
                ),
                "c2_exactly_two_sentences": _is_exactly_two_template_sentences(
                    candidate_text=candidate_c2_text,
                    shared_sentence=shared_sentence,
                    distinguishing_sentence=distinguishing_sentence_c2,
                ),
                "candidate_length_ratio": length_ratio,
                "candidate_length_ratio_in_range": True,
                "no_extra_facts": True,
                "both_candidates_factual_by_source": True,
                "shared_sentence": shared_sentence,
                "distinguishing_template_id": distinguishing_template_id,
            },
        },
    }
    try:
        validate_family_record(family)
    except SchemaValidationError as error:
        raise BioRubricBuildError(str(error)) from error
    return family


def _inspect_biorubric_entity_family(
    entity_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    rows = [dict(row) for row in entity_rows]
    if not rows:
        return None, "empty_entity_rows", {}

    entity_id = str(rows[0]["entity_id"])
    entity_name = str(rows[0]["entity_name"])
    split = str(rows[0]["split"])
    rows_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_slot[str(row["slot"])].append(row)

    shared_rows = _canonicalize_fact_rows(
        row
        for row in rows_by_slot.get(BIORUBRIC_SHARED_SLOT, [])
        if _is_admissible_value_text(str(row["value_text"]))
    )
    if not shared_rows:
        return (
            None,
            "no_shared_occupation",
            {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "split": split,
                "slot_row_counts": {
                    slot: len(slot_rows) for slot, slot_rows in sorted(rows_by_slot.items())
                },
            },
        )

    admissible_slot_counts: dict[str, int] = {}
    for slot in BIORUBRIC_DISTINGUISHING_SLOT_PRIORITY:
        admissible_rows = _canonicalize_fact_rows(
            row
            for row in rows_by_slot.get(slot, [])
            if _is_admissible_value_text(str(row["value_text"]))
        )
        admissible_slot_counts[slot] = len(admissible_rows)
        if len(admissible_rows) < 2:
            continue

        candidate_c1_text = _build_candidate_text(
            entity_name=entity_name,
            shared_value_text=str(shared_rows[0]["value_text"]),
            slot=slot,
            value_text=str(admissible_rows[0]["value_text"]),
        )
        candidate_c2_text = _build_candidate_text(
            entity_name=entity_name,
            shared_value_text=str(shared_rows[0]["value_text"]),
            slot=slot,
            value_text=str(admissible_rows[1]["value_text"]),
        )
        candidate_length_ratio = _candidate_length_ratio(candidate_c1_text, candidate_c2_text)
        if candidate_length_ratio > 1.25:
            return (
                None,
                "candidate_length_ratio_exceeded",
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "split": split,
                    "slot_row_counts": {
                        slot_name: len(slot_rows)
                        for slot_name, slot_rows in sorted(rows_by_slot.items())
                    },
                    "admissible_slot_counts": dict(sorted(admissible_slot_counts.items())),
                    "candidate_length_ratio": candidate_length_ratio,
                    "chosen_distinguishing_slot": slot,
                },
            )

        family = _build_entity_family(rows)
        if family is None:
            raise BioRubricBuildError(
                f"Entity {entity_id!r} satisfied the locked BioRubric rules but did not "
                "produce a family."
            )
        return family, None, {}

    return (
        None,
        "no_distinguishing_slot",
        {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "split": split,
            "slot_row_counts": {
                slot: len(slot_rows) for slot, slot_rows in sorted(rows_by_slot.items())
            },
            "admissible_slot_counts": dict(sorted(admissible_slot_counts.items())),
        },
    )


def _make_biorubric_invalid_record(
    entity_rows: Iterable[Mapping[str, Any]],
    *,
    skip_reason: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in entity_rows]
    entity_id = str(rows[0]["entity_id"])
    entity_name = str(rows[0]["entity_name"])
    split = str(rows[0]["split"])
    return {
        "task_family": "biorubric",
        "family_id": f"biorubric_{entity_id}",
        "entity_id": entity_id,
        "entity_name": entity_name,
        "split": split,
        "skip_reason": skip_reason,
        "source_row_count": len(rows),
        "details": dict(details),
    }


def build_biorubric_families_detailed(
    canonical_rows: Iterable[Mapping[str, Any]],
    split_manifest_rows: Iterable[Mapping[str, Any]],
    *,
    target_family_count_min: int,
    target_family_count_max: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build deterministic BioRubric families plus a QC summary."""

    if target_family_count_min <= 0 or target_family_count_max <= 0:
        raise BioRubricBuildError("BioRubric target family counts must be positive integers.")
    if target_family_count_min > target_family_count_max:
        raise BioRubricBuildError("target_family_count_min must be <= target_family_count_max.")

    split_manifest_by_anchor = _load_biorubric_split_manifest_rows(split_manifest_rows)
    rows_with_split = sorted(
        _attach_splits(canonical_rows, split_manifest_by_anchor),
        key=lambda row: (str(row["entity_id"]), str(row["slot"]), str(row["value_norm"])),
    )

    rows_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_with_split:
        rows_by_entity[str(row["entity_id"])].append(row)

    skip_reasons: Counter[str] = Counter()
    distinguishing_slot_counts: Counter[str] = Counter()
    built_families: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for entity_id in sorted(rows_by_entity):
        entity_rows = rows_by_entity[entity_id]
        family, skip_reason, details = _inspect_biorubric_entity_family(entity_rows)
        if family is None:
            if skip_reason is None:
                raise BioRubricBuildError(
                    f"Entity {entity_id!r} did not produce a family but no skip_reason was set."
                )
            skip_reasons[skip_reason] += 1
            invalid_records.append(
                _make_biorubric_invalid_record(
                    entity_rows,
                    skip_reason=skip_reason,
                    details=details,
                )
            )
            continue

        built_families.append(family)
        distinguishing_slot_counts[str(family["metadata"]["distinguishing_slot"])] += 1

    selected_families = built_families[:target_family_count_max]
    qc_report = {
        "task_family": "biorubric",
        "input_row_count": len(rows_with_split),
        "split_manifest_count": len(split_manifest_by_anchor),
        "built_family_count": len(built_families),
        "selected_family_count": len(selected_families),
        "target_family_count_min": target_family_count_min,
        "target_family_count_max": target_family_count_max,
        "skip_reasons": dict(skip_reasons),
        "selected_split_counts": dict(
            sorted(Counter(family["split"] for family in selected_families).items())
        ),
        "selected_distinguishing_slot_counts": dict(sorted(distinguishing_slot_counts.items())),
        "selection_policy": "sorted_by_family_id_then_truncate_to_max",
    }
    if len(selected_families) < target_family_count_min:
        qc_report["skip_reasons"]["target_family_count_min_not_reached"] = (
            target_family_count_min - len(selected_families)
        )

    return selected_families, qc_report, invalid_records


def build_biorubric_families(
    canonical_rows: Iterable[Mapping[str, Any]],
    split_manifest_rows: Iterable[Mapping[str, Any]],
    *,
    target_family_count_min: int,
    target_family_count_max: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families, qc_report, _ = build_biorubric_families_detailed(
        canonical_rows,
        split_manifest_rows,
        target_family_count_min=target_family_count_min,
        target_family_count_max=target_family_count_max,
    )
    return families, qc_report


def build_biorubric_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the locked BioRubric builder from config and write its artifacts."""

    input_path = config.get("input_path")
    output_path = config.get("output_path")
    target_family_count_min = config.get("target_family_count_min")
    target_family_count_max = config.get("target_family_count_max")
    if not isinstance(input_path, str) or not input_path:
        raise BioRubricBuildError(
            "BioRubric pilot config must contain a non-empty string input_path."
        )
    if not isinstance(output_path, str) or not output_path:
        raise BioRubricBuildError(
            "BioRubric pilot config must contain a non-empty string output_path."
        )
    if not isinstance(target_family_count_min, int) or target_family_count_min <= 0:
        raise BioRubricBuildError("target_family_count_min must be a positive integer.")
    if not isinstance(target_family_count_max, int) or target_family_count_max <= 0:
        raise BioRubricBuildError("target_family_count_max must be a positive integer.")

    split_manifest_path = config.get("split_manifest_path")
    if split_manifest_path is None:
        split_manifest_path = str(derive_sidecar_paths(input_path)["split_manifest_path"])
    if not isinstance(split_manifest_path, str) or not split_manifest_path:
        raise BioRubricBuildError("split_manifest_path must be a non-empty string when present.")

    canonical_rows = load_canonical_facttable(input_path)
    split_manifest_rows = load_jsonl_records(split_manifest_path)
    families, qc_report, _ = build_biorubric_families_detailed(
        canonical_rows,
        split_manifest_rows,
        target_family_count_min=target_family_count_min,
        target_family_count_max=target_family_count_max,
    )

    write_jsonl_records(output_path, families)
    qc_report_path = Path(output_path).with_name(f"{Path(output_path).stem}_qc.json")
    write_json(qc_report_path, qc_report)

    return {
        "output_path": str(output_path),
        "qc_report_path": str(qc_report_path),
        "family_count": len(families),
        "target_family_count_min": target_family_count_min,
        "target_family_count_max": target_family_count_max,
        "split_manifest_path": split_manifest_path,
        "qc_report": qc_report,
    }
