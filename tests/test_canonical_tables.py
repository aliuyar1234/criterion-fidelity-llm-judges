from __future__ import annotations

import pytest

from src.data.canonical_tables import (
    build_canonical_facttable,
    build_canonical_qatable,
    compute_split_assignment,
    load_and_validate_relation_whitelist,
    normalize_text_value,
)
from src.data.constants import BIORUBRIC_ALLOWED_SLOTS


def _qakey_raw_rows() -> list[dict[str, object]]:
    return [
        {
            "subject_id": "Q40",
            "subject_text": "Austria",
            "relation_id": "P36",
            "object_id": "Q1741",
            "object_text": "Vienna",
            "object_aliases": ["Wien"],
            "source_ref": "wikidata:Q40:P36:Q1741",
        },
        {
            "subject_id": "Q90",
            "subject_text": "France",
            "relation_id": "P36",
            "object_id": "Q90_city",
            "object_text": "Paris.",
            "object_aliases": [],
            "source_ref": "wikidata:Q142:P36:Q90",
        },
        {
            "subject_id": "Q419",
            "subject_text": "Peru",
            "relation_id": "P36",
            "object_id": "Q2868",
            "object_text": '"Lima"',
            "object_aliases": [],
            "source_ref": "wikidata:Q419:P36:Q2868",
        },
        {
            "subject_id": "Q999",
            "subject_text": "Atlantis",
            "relation_id": "P36",
            "object_id": "Q1000",
            "object_text": "Poseidonis",
            "object_aliases": [],
            "source_ref": "wikidata:Q999:P36:Q1000",
        },
        {
            "subject_id": "Q999",
            "subject_text": "Atlantis",
            "relation_id": "P36",
            "object_id": "Q1001",
            "object_text": "Cleito City",
            "object_aliases": [],
            "source_ref": "wikidata:Q999:P36:Q1001",
        },
        {
            "subject_id": "Q40",
            "subject_text": "Austria",
            "relation_id": "P31",
            "object_id": "Q3624078",
            "object_text": "sovereign state",
            "object_aliases": [],
            "source_ref": "wikidata:Q40:P31:Q3624078",
        },
    ]


def _biorubric_raw_rows() -> list[dict[str, object]]:
    return [
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "occupation",
            "value_id": "Q170790",
            "value_text": "mathematician",
            "source_ref": "wikidata:Q7259:P106:Q170790",
        },
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "notable_work",
            "value_id": "Q123",
            "value_text": "Notes on the Analytical Engine",
            "source_ref": "wikidata:Q7259:P800:Q123",
        },
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "notable_work",
            "value_id": "Q124",
            "value_text": "Sketch of the Analytical Engine",
            "source_ref": "wikidata:Q7259:P800:Q124",
        },
        {
            "entity_id": "Q42",
            "entity_name": "Douglas Adams",
            "is_human": True,
            "slot": "occupation",
            "value_id": "Q36180",
            "value_text": "writer",
            "source_ref": "wikidata:Q42:P106:Q36180",
        },
        {
            "entity_id": "Q42",
            "entity_name": "Douglas Adams",
            "is_human": True,
            "slot": "award",
            "value_id": "Q185667",
            "value_text": "Hugo Award",
            "source_ref": "wikidata:Q42:P166:Q185667",
        },
        {
            "entity_id": "Q9999",
            "entity_name": "Atlantic Ocean",
            "is_human": False,
            "slot": "field",
            "value_id": "Q100",
            "value_text": "oceanography",
            "source_ref": "wikidata:Q9999:P101:Q100",
        },
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "spouse",
            "value_id": "Q111",
            "value_text": "William King-Noel",
            "source_ref": "wikidata:Q7259:P26:Q111",
        },
    ]


def test_normalize_text_value_matches_locked_rules() -> None:
    assert normalize_text_value('  "Lima"  ') == "lima"
    assert normalize_text_value("New   York!") == "new york"


def test_build_canonical_qatable_filters_and_renders_questions() -> None:
    whitelist = load_and_validate_relation_whitelist(
        {
            "relation_whitelist": [
                {
                    "relation_id": relation_id,
                    "relation_name": spec["relation_name"],
                    "coarse_type": spec["coarse_type"],
                    "question_template": spec["question_template"],
                }
                for relation_id, spec in {
                    "P19": {
                        "relation_name": "place_of_birth",
                        "coarse_type": "location",
                        "question_template": "Where was {SUBJECT} born?",
                    },
                    "P20": {
                        "relation_name": "place_of_death",
                        "coarse_type": "location",
                        "question_template": "Where did {SUBJECT} die?",
                    },
                    "P569": {
                        "relation_name": "date_of_birth",
                        "coarse_type": "date",
                        "question_template": "When was {SUBJECT} born?",
                    },
                    "P570": {
                        "relation_name": "date_of_death",
                        "coarse_type": "date",
                        "question_template": "When did {SUBJECT} die?",
                    },
                    "P36": {
                        "relation_name": "capital",
                        "coarse_type": "location",
                        "question_template": "What is the capital of {SUBJECT}?",
                    },
                    "P38": {
                        "relation_name": "currency",
                        "coarse_type": "currency",
                        "question_template": "What currency is used in {SUBJECT}?",
                    },
                }.items()
            ]
        }
    )
    rows, split_manifest, qc_report = build_canonical_qatable(_qakey_raw_rows(), whitelist)

    assert len(rows) == 3
    assert qc_report["skip_reasons"]["multi_gold_object"] == 1
    assert qc_report["skip_reasons"]["relation_not_whitelisted"] == 1
    assert rows[0]["question"] == "What is the capital of Austria?"
    assert rows[1]["answer_norm"] == "lima"
    assert len(split_manifest) == 3
    assert (
        split_manifest[0]["split"]
        == compute_split_assignment(split_manifest[0]["qa_anchor_id"])["split"]
    )


def test_build_canonical_facttable_reports_slot_coverage() -> None:
    rows, split_manifest, qc_report = build_canonical_facttable(
        _biorubric_raw_rows(),
        BIORUBRIC_ALLOWED_SLOTS,
    )

    assert len(rows) == 5
    assert qc_report["skip_reasons"]["non_human_entity"] == 1
    assert qc_report["skip_reasons"]["disallowed_slot"] == 1
    assert qc_report["slot_row_counts"]["occupation"] == 2
    assert qc_report["slot_entity_counts"]["occupation"] == 2
    assert len(split_manifest) == 2


def test_relation_whitelist_rejects_drift() -> None:
    with pytest.raises(ValueError):
        load_and_validate_relation_whitelist(
            {
                "relation_whitelist": [
                    {
                        "relation_id": "P36",
                        "relation_name": "capital",
                        "coarse_type": "date",
                        "question_template": "What is the capital of {SUBJECT}?",
                    }
                ]
            }
        )
