from __future__ import annotations

import pytest

from src.data.wikidata_qakey_source import (
    WikidataQAKeySourceError,
    _extract_entity_id,
    _format_wikidata_time_value,
    _rows_from_subject_payload,
    _search_query_for_relation,
    _sparql_subject_id_query,
)


def test_extract_entity_id_requires_wikidata_entity_url() -> None:
    assert _extract_entity_id("http://www.wikidata.org/entity/Q42") == "Q42"
    assert _extract_entity_id("https://www.wikidata.org/entity/Q90") == "Q90"

    with pytest.raises(WikidataQAKeySourceError):
        _extract_entity_id("https://example.com/entity/Q42")


def test_search_query_contains_locked_relation_and_class_filters() -> None:
    query = _search_query_for_relation("P36", "Q3624078")

    assert query == "haswbstatement:P31=Q3624078 haswbstatement:P36"


def test_sparql_subject_query_contains_locked_relation_class_and_pagination() -> None:
    query = _sparql_subject_id_query("P569", subject_class_qid="Q5", limit=200, offset=400)

    assert "wdt:P31 wd:Q5" in query
    assert "wdt:P569 ?object" in query
    assert "LIMIT 200" in query
    assert "OFFSET 400" in query


def test_format_wikidata_time_value_uses_precision_sensitive_iso_strings() -> None:
    assert (
        _format_wikidata_time_value({"time": "+1952-04-11T00:00:00Z", "precision": 11})
        == "1952-04-11"
    )
    assert (
        _format_wikidata_time_value({"time": "+1952-04-00T00:00:00Z", "precision": 10}) == "1952-04"
    )
    assert _format_wikidata_time_value({"time": "+1952-00-00T00:00:00Z", "precision": 9}) == "1952"


def test_rows_from_subject_payload_maps_entity_and_time_relations() -> None:
    subject_payload = {
        "labels": {"en": {"value": "Ada Lovelace"}},
        "claims": {
            "P19": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q84"}},
                    },
                }
            ],
            "P569": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"time": "+1815-12-10T00:00:00Z", "precision": 11}},
                    },
                }
            ],
        },
    }

    place_rows = _rows_from_subject_payload(
        "Q7259",
        "P19",
        subject_payload,
        object_labels={"Q84": "London"},
    )
    date_rows = _rows_from_subject_payload(
        "Q7259",
        "P569",
        subject_payload,
        object_labels={},
    )

    assert place_rows == [
        {
            "subject_id": "Q7259",
            "subject_text": "Ada Lovelace",
            "relation_id": "P19",
            "object_id": "Q84",
            "object_text": "London",
            "object_aliases": [],
            "source_ref": "wikidata:Q7259:P19:Q84",
        }
    ]
    assert date_rows == [
        {
            "subject_id": "Q7259",
            "subject_text": "Ada Lovelace",
            "relation_id": "P569",
            "object_id": None,
            "object_text": "1815-12-10",
            "object_aliases": [],
            "source_ref": "wikidata:Q7259:P569:1815-12-10",
        }
    ]
