from __future__ import annotations

from src.data.qakey_builder import build_qakey_families, select_qakey_donor
from src.data.text_normalization import normalize_text_v1, token_count_v1


def _qa_row(
    *,
    qa_id: str,
    subject_id: str,
    subject_text: str,
    answer_id: str | None,
    answer_text: str,
    aliases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "qa_id": qa_id,
        "question": f"What is the capital of {subject_text}?",
        "subject_id": subject_id,
        "subject_text": subject_text,
        "relation_id": "P36",
        "relation_name": "capital",
        "answer_id": answer_id,
        "answer_text": answer_text,
        "answer_norm": normalize_text_v1(answer_text),
        "answer_aliases_norm": [normalize_text_v1(alias) for alias in (aliases or [])],
        "coarse_type": "location",
        "source_ref": f"wikidata:{qa_id}",
    }


def test_normalize_and_token_count_follow_locked_v1_rules() -> None:
    assert normalize_text_v1('  "Lima"  ') == "lima"
    assert normalize_text_v1("New\tYork!") == "new york"
    assert token_count_v1("New\tYork") == 2
    assert token_count_v1("   ") == 0


def test_select_qakey_donor_uses_lexicographic_tiebreak_after_length_checks() -> None:
    anchor = _qa_row(
        qa_id="Q40:P36",
        subject_id="Q40",
        subject_text="Austria",
        answer_id="Q1741",
        answer_text="Vienna",
        aliases=["Wien"],
    )
    donors = [
        _qa_row(
            qa_id="Q39:P36",
            subject_id="Q39",
            subject_text="Switzerland",
            answer_id="Q70",
            answer_text="Milan",
        ),
        _qa_row(
            qa_id="Q38:P36",
            subject_id="Q38",
            subject_text="Italy",
            answer_id="Q220",
            answer_text="Paris",
        ),
    ]
    for donor in donors:
        donor["split"] = "train"
    anchor["split"] = "train"

    chosen_donor, donor_evaluations = select_qakey_donor(anchor, donors)

    assert chosen_donor is not None
    assert chosen_donor["answer_norm"] == "milan"
    assert all(evaluation["checks"]["is_admissible"] for evaluation in donor_evaluations)


def test_select_qakey_donor_uses_object_id_as_final_tiebreaker() -> None:
    anchor = _qa_row(
        qa_id="Q40:P36",
        subject_id="Q40",
        subject_text="Austria",
        answer_id="Q1741",
        answer_text="Vienna",
    )
    donors = [
        _qa_row(
            qa_id="Q39:P36",
            subject_id="Q39",
            subject_text="Donor One",
            answer_id="Q900",
            answer_text="Paris",
        ),
        _qa_row(
            qa_id="Q38:P36",
            subject_id="Q38",
            subject_text="Donor Two",
            answer_id="Q100",
            answer_text="Paris",
        ),
    ]
    for donor in donors:
        donor["split"] = "train"
    anchor["split"] = "train"

    chosen_donor, _ = select_qakey_donor(anchor, donors)

    assert chosen_donor is not None
    assert chosen_donor["answer_id"] == "Q100"


def test_build_qakey_families_respects_split_sidecars_and_ratio_threshold() -> None:
    canonical_rows = [
        _qa_row(
            qa_id="Q40:P36",
            subject_id="Q40",
            subject_text="Austria",
            answer_id="Q1741",
            answer_text="Vienna",
            aliases=["Wien"],
        ),
        _qa_row(
            qa_id="Q419:P36",
            subject_id="Q419",
            subject_text="Peru",
            answer_id="Q2868",
            answer_text="Lima",
        ),
        _qa_row(
            qa_id="Q142:P36",
            subject_id="Q142",
            subject_text="France",
            answer_id="Q90",
            answer_text="Paris",
        ),
    ]
    split_manifest = [
        {"qa_anchor_id": "Q40", "split": "train", "bucket": 4},
        {"qa_anchor_id": "Q419", "split": "train", "bucket": 13},
        {"qa_anchor_id": "Q142", "split": "test", "bucket": 17},
    ]

    families, qc_report = build_qakey_families(
        canonical_rows,
        split_manifest,
        target_family_count=200,
    )

    assert len(families) == 1
    family = families[0]
    assert family["family_id"] == "qakey_Q419_P36"
    assert family["split"] == "train"
    assert family["candidates"][0]["text"] == "The answer is Lima."
    assert family["candidates"][1]["text"] == "The answer is Vienna."
    assert family["metadata"]["selected_donor_checks"]["char_length_ratio_in_range"] is True
    assert qc_report["built_family_count"] == 1
    assert qc_report["skip_reasons"]["no_admissible_donor"] == 2
    assert qc_report["skip_reasons"]["target_family_count_not_reached"] == 199
    assert qc_report["donor_failure_counts"]["char_length_ratio_in_range"] >= 1
