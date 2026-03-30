from __future__ import annotations

from src.data.wikidata_biorubric_source import (
    WikidataBioRubricSourceError,
    _extract_entity_id,
    _extract_truthy_entity_ids_from_claims,
    _rows_from_entity_payload,
    _search_query_for_slot,
    _sparql_entity_id_query,
)


def test_extract_entity_id_requires_wikidata_entity_url() -> None:
    assert _extract_entity_id("http://www.wikidata.org/entity/Q42") == "Q42"
    assert _extract_entity_id("https://www.wikidata.org/entity/Q90") == "Q90"

    try:
        _extract_entity_id("https://example.com/entity/Q42")
    except WikidataBioRubricSourceError:
        pass
    else:  # pragma: no cover - defensive sanity check
        raise AssertionError("Expected WikidataBioRubricSourceError for non-Wikidata URL.")


def test_search_query_uses_human_and_occupation_filters() -> None:
    query = _search_query_for_slot("P166")

    assert query == "haswbstatement:P31=Q5 haswbstatement:P106 haswbstatement:P166"


def test_sparql_entity_query_uses_human_occupation_slot_and_pagination() -> None:
    query = _sparql_entity_id_query("award", limit=200, offset=600)

    assert "wdt:P31 wd:Q5" in query
    assert "wdt:P106 ?occupation" in query
    assert "wdt:P166 ?value" in query
    assert "LIMIT 200" in query
    assert "OFFSET 600" in query


def test_extract_truthy_entity_ids_prefers_preferred_rank() -> None:
    claims = {
        "P166": [
            {
                "rank": "normal",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"value": {"id": "Q1"}},
                },
            },
            {
                "rank": "preferred",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"value": {"id": "Q2"}},
                },
            },
            {
                "rank": "preferred",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"value": {"numeric-id": 3}},
                },
            },
            {
                "rank": "deprecated",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"value": {"id": "Q999"}},
                },
            },
        ]
    }

    assert _extract_truthy_entity_ids_from_claims(claims, "P166") == ["Q2", "Q3"]


def test_rows_from_entity_payload_keeps_only_human_entities_with_eligible_slots() -> None:
    entity_payload = {
        "labels": {"en": {"value": "Ada Lovelace"}},
        "claims": {
            "P31": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q5"}},
                    },
                }
            ],
            "P106": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q170790"}},
                    },
                }
            ],
            "P166": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q2"}},
                    },
                },
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q3"}},
                    },
                },
            ],
            "P101": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q4"}},
                    },
                }
            ],
        },
    }

    rows, eligible_slots = _rows_from_entity_payload(
        "Q7259",
        entity_payload,
        value_labels={
            "Q170790": "mathematician",
            "Q2": "Royal Medal",
            "Q3": "Turing Award",
            "Q4": "computer science",
        },
    )

    assert eligible_slots == {"award"}
    assert rows == [
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "award",
            "value_id": "Q2",
            "value_text": "Royal Medal",
            "source_ref": "wikidata:Q7259:P166:Q2",
        },
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "award",
            "value_id": "Q3",
            "value_text": "Turing Award",
            "source_ref": "wikidata:Q7259:P166:Q3",
        },
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "field",
            "value_id": "Q4",
            "value_text": "computer science",
            "source_ref": "wikidata:Q7259:P101:Q4",
        },
        {
            "entity_id": "Q7259",
            "entity_name": "Ada Lovelace",
            "is_human": True,
            "slot": "occupation",
            "value_id": "Q170790",
            "value_text": "mathematician",
            "source_ref": "wikidata:Q7259:P106:Q170790",
        },
    ]
