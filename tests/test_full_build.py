from __future__ import annotations

import json

import pytest

from src.data.full_build import (
    FullBuildError,
    _freeze_families_by_split,
    _load_existing_source_rows_if_matching_fetch_config,
)


def _family(task_family: str, split: str, family_id: str) -> dict[str, object]:
    return {"task_family": task_family, "split": split, "family_id": family_id}


def test_freeze_families_by_split_applies_per_split_targets_deterministically() -> None:
    built_families = [
        _family("qa_key", "train", "qakey_c"),
        _family("qa_key", "train", "qakey_a"),
        _family("qa_key", "train", "qakey_b"),
        _family("qa_key", "dev", "qakey_e"),
        _family("qa_key", "dev", "qakey_d"),
        _family("qa_key", "test", "qakey_f"),
    ]

    selected_by_split, discarded_records, selection_summary = _freeze_families_by_split(
        "qa_key",
        built_families,
        targets={"train": 2, "dev": 1, "test": 1},
    )

    assert [family["family_id"] for family in selected_by_split["train"]] == [
        "qakey_a",
        "qakey_b",
    ]
    assert [family["family_id"] for family in selected_by_split["dev"]] == ["qakey_d"]
    assert [family["family_id"] for family in selected_by_split["test"]] == ["qakey_f"]
    assert discarded_records == [
        {
            "task_family": "qa_key",
            "family_id": "qakey_c",
            "split": "train",
            "discard_reason": "split_quota_exceeded",
        },
        {
            "task_family": "qa_key",
            "family_id": "qakey_e",
            "split": "dev",
            "discard_reason": "split_quota_exceeded",
        },
    ]
    assert selection_summary["selected_split_counts"] == {"train": 2, "dev": 1, "test": 1}


def test_freeze_families_by_split_raises_on_shortfall() -> None:
    built_families = [
        _family("biorubric", "train", "biorubric_a"),
        _family("biorubric", "dev", "biorubric_b"),
    ]

    with pytest.raises(FullBuildError):
        _freeze_families_by_split(
            "biorubric",
            built_families,
            targets={"train": 1, "dev": 1, "test": 1},
        )


def test_load_existing_source_rows_if_matching_fetch_config_reuses_matching_artifacts(
    tmp_path,
) -> None:
    output_path = tmp_path / "source_rows.jsonl"
    summary_path = tmp_path / "source_rows_fetch_summary.json"
    output_path.write_text(
        json.dumps({"subject_id": "Q1", "relation_id": "P19"}) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "stage": "raw_qakey_wikidata_fetch",
                "relation_requests": [
                    {
                        "relation_id": "P19",
                        "subject_class_qid": "Q5",
                        "page_size_requested": 50,
                        "max_pages": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reused = _load_existing_source_rows_if_matching_fetch_config(
        fetch_config={
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "relation_requests": [
                {
                    "relation_id": "P19",
                    "subject_class_qid": "Q5",
                    "page_size": 50,
                    "max_pages": 60,
                }
            ],
        },
        request_key="relation_requests",
        summary_request_id_key="relation_id",
        stage_name="raw_qakey_wikidata_fetch",
    )

    assert reused is not None
    rows, summary = reused
    assert rows == [{"subject_id": "Q1", "relation_id": "P19"}]
    assert summary["stage"] == "raw_qakey_wikidata_fetch"


def test_load_existing_source_rows_if_matching_fetch_config_rejects_mismatched_summary(
    tmp_path,
) -> None:
    output_path = tmp_path / "truthy_human_facts.jsonl"
    summary_path = tmp_path / "truthy_human_facts_fetch_summary.json"
    output_path.write_text(json.dumps({"entity_id": "Q1"}) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "stage": "raw_biorubric_wikidata_fetch",
                "slot_requests": [
                    {
                        "slot": "award",
                        "page_size_requested": 25,
                        "max_pages": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reused = _load_existing_source_rows_if_matching_fetch_config(
        fetch_config={
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "slot_requests": [{"slot": "award", "page_size": 50, "max_pages": 10}],
        },
        request_key="slot_requests",
        summary_request_id_key="slot",
        stage_name="raw_biorubric_wikidata_fetch",
    )

    assert reused is None
