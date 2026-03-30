from __future__ import annotations

from src.data.biorubric_builder import build_biorubric_families
from src.data.text_normalization import normalize_text_v1


def _fact(
    *,
    entity_id: str,
    entity_name: str,
    slot: str,
    value_id: str,
    value_text: str,
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "slot": slot,
        "value_id": value_id,
        "value_text": value_text,
        "value_norm": normalize_text_v1(value_text),
        "source_ref": f"wikidata:{entity_id}:{slot}:{value_id}",
    }


def test_build_biorubric_families_uses_locked_slot_priority_and_templates() -> None:
    canonical_rows = [
        _fact(
            entity_id="Q1",
            entity_name="Ada Lovelace",
            slot="occupation",
            value_id="Q170790",
            value_text="mathematician",
        ),
        _fact(
            entity_id="Q1",
            entity_name="Ada Lovelace",
            slot="award",
            value_id="Q2",
            value_text="Royal Medal",
        ),
        _fact(
            entity_id="Q1",
            entity_name="Ada Lovelace",
            slot="award",
            value_id="Q3",
            value_text="Turing Award",
        ),
        _fact(
            entity_id="Q1",
            entity_name="Ada Lovelace",
            slot="field",
            value_id="Q4",
            value_text="computer science",
        ),
        _fact(
            entity_id="Q1",
            entity_name="Ada Lovelace",
            slot="field",
            value_id="Q5",
            value_text="mathematics",
        ),
        _fact(
            entity_id="Q2",
            entity_name="Skipped Example",
            slot="occupation",
            value_id="Q10",
            value_text="writer",
        ),
        _fact(
            entity_id="Q2",
            entity_name="Skipped Example",
            slot="field",
            value_id="Q11",
            value_text="poetry",
        ),
    ]
    split_manifest = [
        {"entity_anchor_id": "Q1", "split": "dev", "bucket": 14},
        {"entity_anchor_id": "Q2", "split": "train", "bucket": 3},
    ]

    families, qc_report = build_biorubric_families(
        canonical_rows,
        split_manifest,
        target_family_count_min=100,
        target_family_count_max=150,
    )

    assert len(families) == 1
    family = families[0]
    assert family["family_id"] == "biorubric_Q1"
    assert family["split"] == "dev"
    assert family["metadata"]["distinguishing_slot"] == "award"
    assert family["candidates"][0]["text"] == (
        "Ada Lovelace was a mathematician. Ada Lovelace received Royal Medal."
    )
    assert family["candidates"][1]["text"] == (
        "Ada Lovelace was a mathematician. Ada Lovelace received Turing Award."
    )
    assert family["variants"][1]["variant_id"] == "para_lex"
    assert family["variants"][1]["criterion_text"].startswith(
        "Give 1 point(s) for stating that Ada Lovelace was a mathematician."
    )
    assert (
        family["variants"][2]["criterion_text"]
        .splitlines()[0]
        .startswith("Award 2 point(s) if the biography mentions Royal Medal.")
    )
    assert qc_report["skip_reasons"]["no_distinguishing_slot"] == 1


def test_build_biorubric_families_enforces_candidate_length_ratio() -> None:
    canonical_rows = [
        _fact(
            entity_id="Q3",
            entity_name="Ratio Example",
            slot="occupation",
            value_id="Q20",
            value_text="writer",
        ),
        _fact(
            entity_id="Q3",
            entity_name="Ratio Example",
            slot="field",
            value_id="Q21",
            value_text="art",
        ),
        _fact(
            entity_id="Q3",
            entity_name="Ratio Example",
            slot="field",
            value_id="Q22",
            value_text=("history theory criticism sociology economics politics"),
        ),
    ]
    split_manifest = [{"entity_anchor_id": "Q3", "split": "test", "bucket": 18}]

    families, qc_report = build_biorubric_families(
        canonical_rows,
        split_manifest,
        target_family_count_min=100,
        target_family_count_max=150,
    )

    assert families == []
    assert qc_report["skip_reasons"]["candidate_length_ratio_exceeded"] == 1


def test_build_biorubric_families_do_not_miscount_initials_as_extra_sentences() -> None:
    canonical_rows = [
        _fact(
            entity_id="Q4",
            entity_name="Robert J. Hirosky",
            slot="occupation",
            value_id="Q20",
            value_text="physicist",
        ),
        _fact(
            entity_id="Q4",
            entity_name="Robert J. Hirosky",
            slot="field",
            value_id="Q21",
            value_text="quantum information",
        ),
        _fact(
            entity_id="Q4",
            entity_name="Robert J. Hirosky",
            slot="field",
            value_id="Q22",
            value_text="quantum mechanics",
        ),
    ]
    split_manifest = [{"entity_anchor_id": "Q4", "split": "dev", "bucket": 14}]

    families, _ = build_biorubric_families(
        canonical_rows,
        split_manifest,
        target_family_count_min=100,
        target_family_count_max=150,
    )

    assert len(families) == 1
    checks = families[0]["metadata"]["candidate_checks"]
    assert checks["c1_exactly_two_sentences"] is True
    assert checks["c2_exactly_two_sentences"] is True
