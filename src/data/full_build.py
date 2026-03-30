from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from src.common.config import load_config
from src.data.biorubric_builder import build_biorubric_families_detailed
from src.data.canonical_tables import (
    build_canonical_facttable,
    build_canonical_qatable,
    derive_sidecar_paths,
    load_and_validate_relation_whitelist,
    load_jsonl_records,
    write_json,
    write_jsonl_records,
)
from src.data.qakey_builder import build_qakey_families_detailed
from src.data.wikidata_biorubric_source import (
    fetch_biorubric_source_rows_from_config,
    write_biorubric_source_artifacts,
)
from src.data.wikidata_qakey_source import (
    fetch_qakey_source_rows_from_config,
    write_qakey_source_artifacts,
)

TASK_FAMILY_DIRS = {"qa_key": "qakey", "biorubric": "biorubric"}
FROZEN_SPLIT_ORDER = ("train", "dev", "test")


class FullBuildError(ValueError):
    """Raised when the M5 full-dataset freeze cannot be completed honestly."""


def _validate_targets(targets: Mapping[str, Any]) -> dict[str, int]:
    validated: dict[str, int] = {}
    for split in FROZEN_SPLIT_ORDER:
        value = targets.get(split)
        if not isinstance(value, int) or value <= 0:
            raise FullBuildError(
                f"full_build targets must contain positive integer count for split {split!r}."
            )
        validated[split] = value
    return validated


def _load_nested_config(config: Mapping[str, Any], path_key: str) -> dict[str, Any]:
    path = config.get(path_key)
    if not isinstance(path, str) or not path:
        raise FullBuildError(f"full_build config must contain non-empty string {path_key!r}.")
    return load_config(path)


def _load_existing_source_rows_if_matching_fetch_config(
    *,
    fetch_config: Mapping[str, Any],
    request_key: str,
    summary_request_id_key: str,
    stage_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    output_path = fetch_config.get("output_path")
    summary_path = fetch_config.get("summary_path")
    if not isinstance(output_path, str) or not isinstance(summary_path, str):
        return None

    output_file = Path(output_path)
    summary_file = Path(summary_path)
    if not output_file.exists() or not summary_file.exists():
        return None

    summary = load_config(summary_path)
    if summary.get("stage") != stage_name:
        return None

    config_requests = fetch_config.get(request_key)
    summary_requests = summary.get(request_key)
    if not isinstance(config_requests, list) or not isinstance(summary_requests, list):
        return None

    normalized_summary_requests: dict[str, tuple[int | None, int | None, str | None]] = {}
    for summary_request in summary_requests:
        if not isinstance(summary_request, Mapping):
            return None
        request_id = summary_request.get(summary_request_id_key)
        if not isinstance(request_id, str):
            return None
        normalized_summary_requests[request_id] = (
            summary_request.get("page_size_requested", summary_request.get("page_size")),
            summary_request.get("max_pages"),
            summary_request.get("subject_class_qid")
            if summary_request_id_key == "relation_id"
            else None,
        )

    for config_request in config_requests:
        if not isinstance(config_request, Mapping):
            return None
        request_id = config_request.get(summary_request_id_key)
        if not isinstance(request_id, str):
            return None
        if request_id not in normalized_summary_requests:
            return None

        summary_page_size, summary_max_pages, summary_subject_class = normalized_summary_requests[
            request_id
        ]
        if summary_page_size != config_request.get("page_size"):
            return None
        if summary_max_pages != config_request.get("max_pages"):
            return None
        if summary_request_id_key == "relation_id" and summary_subject_class != config_request.get(
            "subject_class_qid"
        ):
            return None

    return load_jsonl_records(output_path), summary


def _task_output_paths(task_family: str) -> dict[str, Path]:
    task_dir = Path("data/processed") / TASK_FAMILY_DIRS[task_family]
    return {
        "train_path": task_dir / "train_families.jsonl",
        "dev_path": task_dir / "dev_families.jsonl",
        "test_path": task_dir / "test_families.jsonl",
        "combined_path": task_dir / "full_families.jsonl",
        "split_manifest_path": task_dir / "full_split_manifest.jsonl",
        "qc_report_path": task_dir / "full_freeze_qc.json",
        "invalid_log_path": task_dir / "full_invalid_families.jsonl",
        "discarded_log_path": task_dir / "full_discarded_families.jsonl",
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _freeze_families_by_split(
    task_family: str,
    built_families: list[dict[str, Any]],
    *,
    targets: Mapping[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    families_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in built_families:
        families_by_split[str(family["split"])].append(dict(family))

    for split in FROZEN_SPLIT_ORDER:
        families_by_split[split] = sorted(
            families_by_split.get(split, []),
            key=lambda family: str(family["family_id"]),
        )

    available_counts = {split: len(families_by_split[split]) for split in FROZEN_SPLIT_ORDER}
    shortfalls = {
        split: targets[split] - available_counts[split]
        for split in FROZEN_SPLIT_ORDER
        if available_counts[split] < targets[split]
    }
    if shortfalls:
        raise FullBuildError(
            f"{task_family} full freeze cannot meet split targets with current built families: "
            f"available={available_counts}, targets={dict(targets)}, shortfalls={shortfalls}"
        )

    selected_by_split = {
        split: families_by_split[split][: targets[split]] for split in FROZEN_SPLIT_ORDER
    }
    discarded_records: list[dict[str, Any]] = []
    for split in FROZEN_SPLIT_ORDER:
        for family in families_by_split[split][targets[split] :]:
            discarded_records.append(
                {
                    "task_family": task_family,
                    "family_id": family["family_id"],
                    "split": split,
                    "discard_reason": "split_quota_exceeded",
                }
            )

    selection_summary = {
        "available_split_counts": available_counts,
        "selected_split_counts": {
            split: len(selected_by_split[split]) for split in FROZEN_SPLIT_ORDER
        },
        "discarded_split_counts": dict(
            sorted(Counter(record["split"] for record in discarded_records).items())
        ),
        "selection_policy": "sorted_by_family_id_with_per_split_targets",
    }
    return selected_by_split, discarded_records, selection_summary


def _write_frozen_family_outputs(
    task_family: str,
    selected_by_split: Mapping[str, list[dict[str, Any]]],
    invalid_records: list[dict[str, Any]],
    discarded_records: list[dict[str, Any]],
    qc_report: Mapping[str, Any],
) -> dict[str, str]:
    output_paths = _task_output_paths(task_family)
    combined_records: list[dict[str, Any]] = []
    split_manifest_rows: list[dict[str, Any]] = []

    for split in FROZEN_SPLIT_ORDER:
        records = list(selected_by_split[split])
        write_jsonl_records(output_paths[f"{split}_path"], records)
        combined_records.extend(records)
        split_manifest_rows.extend(
            {"family_id": record["family_id"], "split": split} for record in records
        )

    write_jsonl_records(output_paths["combined_path"], combined_records)
    write_jsonl_records(output_paths["split_manifest_path"], split_manifest_rows)
    write_jsonl_records(output_paths["invalid_log_path"], invalid_records)
    write_jsonl_records(output_paths["discarded_log_path"], discarded_records)
    write_json(output_paths["qc_report_path"], qc_report)

    return {key: str(path) for key, path in output_paths.items()}


def _build_qakey_canonical_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    relation_whitelist = load_and_validate_relation_whitelist(config)
    raw_rows = load_jsonl_records(config["input_path"])
    canonical_rows, split_manifest, qc_report = build_canonical_qatable(
        raw_rows=raw_rows,
        relation_whitelist=relation_whitelist,
    )
    output_path = Path(config["output_path"])
    sidecar_paths = derive_sidecar_paths(output_path)
    write_jsonl_records(output_path, canonical_rows)
    write_jsonl_records(sidecar_paths["split_manifest_path"], split_manifest)
    write_json(sidecar_paths["qc_report_path"], qc_report)
    return {
        "output_path": str(output_path),
        "split_manifest_path": str(sidecar_paths["split_manifest_path"]),
        "qc_report_path": str(sidecar_paths["qc_report_path"]),
        "output_row_count": len(canonical_rows),
        "split_anchor_count": len(split_manifest),
        "skip_reasons": qc_report["skip_reasons"],
        "relation_output_counts": qc_report["relation_output_counts"],
    }


def _build_biorubric_canonical_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_rows = load_jsonl_records(config["input_path"])
    canonical_rows, split_manifest, qc_report = build_canonical_facttable(
        raw_rows=raw_rows,
        allowed_slots=config["allowed_slots"],
    )
    output_path = Path(config["output_path"])
    sidecar_paths = derive_sidecar_paths(output_path)
    write_jsonl_records(output_path, canonical_rows)
    write_jsonl_records(sidecar_paths["split_manifest_path"], split_manifest)
    write_json(sidecar_paths["qc_report_path"], qc_report)
    return {
        "output_path": str(output_path),
        "split_manifest_path": str(sidecar_paths["split_manifest_path"]),
        "qc_report_path": str(sidecar_paths["qc_report_path"]),
        "output_row_count": len(canonical_rows),
        "split_anchor_count": len(split_manifest),
        "skip_reasons": qc_report["skip_reasons"],
        "slot_row_counts": qc_report["slot_row_counts"],
    }


def _build_qakey_full_freeze(
    *,
    fetch_config: Mapping[str, Any],
    canonical_config: Mapping[str, Any],
    targets: Mapping[str, int],
) -> dict[str, Any]:
    existing_source = _load_existing_source_rows_if_matching_fetch_config(
        fetch_config=fetch_config,
        request_key="relation_requests",
        summary_request_id_key="relation_id",
        stage_name="raw_qakey_wikidata_fetch",
    )
    if existing_source is None:
        raw_rows, raw_summary = fetch_qakey_source_rows_from_config(fetch_config)
        write_qakey_source_artifacts(
            output_path=fetch_config["output_path"],
            summary_path=fetch_config["summary_path"],
            rows=raw_rows,
            summary=raw_summary,
        )
    else:
        raw_rows, raw_summary = existing_source

    canonical_result = _build_qakey_canonical_from_config(canonical_config)
    canonical_rows = load_jsonl_records(canonical_config["output_path"])
    split_manifest_rows = load_jsonl_records(canonical_result["split_manifest_path"])
    built_families, builder_qc, invalid_records = build_qakey_families_detailed(
        canonical_rows,
        split_manifest_rows,
        target_family_count=None,
    )
    selected_by_split, discarded_records, selection_summary = _freeze_families_by_split(
        "qa_key",
        built_families,
        targets=targets,
    )

    combined_families = [
        family for split in FROZEN_SPLIT_ORDER for family in selected_by_split[split]
    ]
    qc_report = {
        "task_family": "qa_key",
        "freeze_stage": "m5_full_dataset_freeze",
        "raw_source_summary": raw_summary,
        "canonical_summary": canonical_result,
        "builder_summary": builder_qc,
        "selection_summary": selection_summary,
        "invalid_family_count": len(invalid_records),
        "discarded_family_count": len(discarded_records),
        "selected_relation_counts": dict(
            sorted(
                Counter(family["metadata"]["relation_id"] for family in combined_families).items()
            )
        ),
    }
    output_paths = _write_frozen_family_outputs(
        "qa_key",
        selected_by_split,
        invalid_records,
        discarded_records,
        qc_report,
    )
    return {
        "task_family": "qa_key",
        "raw_source_summary": raw_summary,
        "canonical_summary": canonical_result,
        "builder_summary": builder_qc,
        "selection_summary": selection_summary,
        "output_paths": output_paths,
    }


def _build_biorubric_full_freeze(
    *,
    fetch_config: Mapping[str, Any],
    canonical_config: Mapping[str, Any],
    targets: Mapping[str, int],
) -> dict[str, Any]:
    existing_source = _load_existing_source_rows_if_matching_fetch_config(
        fetch_config=fetch_config,
        request_key="slot_requests",
        summary_request_id_key="slot",
        stage_name="raw_biorubric_wikidata_fetch",
    )
    if existing_source is None:
        raw_rows, raw_summary = fetch_biorubric_source_rows_from_config(fetch_config)
        write_biorubric_source_artifacts(
            output_path=fetch_config["output_path"],
            summary_path=fetch_config["summary_path"],
            rows=raw_rows,
            summary=raw_summary,
        )
    else:
        raw_rows, raw_summary = existing_source

    canonical_result = _build_biorubric_canonical_from_config(canonical_config)
    canonical_rows = load_jsonl_records(canonical_config["output_path"])
    split_manifest_rows = load_jsonl_records(canonical_result["split_manifest_path"])
    built_families, builder_qc, invalid_records = build_biorubric_families_detailed(
        canonical_rows,
        split_manifest_rows,
        target_family_count_min=1,
        target_family_count_max=max(len(canonical_rows), 1_000_000),
    )
    selected_by_split, discarded_records, selection_summary = _freeze_families_by_split(
        "biorubric",
        built_families,
        targets=targets,
    )

    combined_families = [
        family for split in FROZEN_SPLIT_ORDER for family in selected_by_split[split]
    ]
    qc_report = {
        "task_family": "biorubric",
        "freeze_stage": "m5_full_dataset_freeze",
        "raw_source_summary": raw_summary,
        "canonical_summary": canonical_result,
        "builder_summary": builder_qc,
        "selection_summary": selection_summary,
        "invalid_family_count": len(invalid_records),
        "discarded_family_count": len(discarded_records),
        "selected_distinguishing_slot_counts": dict(
            sorted(
                Counter(
                    family["metadata"]["distinguishing_slot"] for family in combined_families
                ).items()
            )
        ),
    }
    output_paths = _write_frozen_family_outputs(
        "biorubric",
        selected_by_split,
        invalid_records,
        discarded_records,
        qc_report,
    )
    return {
        "task_family": "biorubric",
        "raw_source_summary": raw_summary,
        "canonical_summary": canonical_result,
        "builder_summary": builder_qc,
        "selection_summary": selection_summary,
        "output_paths": output_paths,
    }


def _build_dataset_stats_rows(task_summaries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in task_summaries:
        task_family = str(summary["task_family"])
        qc_report_path = summary["output_paths"]["qc_report_path"]
        qc_report = load_config(qc_report_path)
        invalid_family_count = int(qc_report["invalid_family_count"])
        discarded_family_count = int(qc_report["discarded_family_count"])
        selection_summary = qc_report["selection_summary"]

        for split in FROZEN_SPLIT_ORDER:
            family_count = int(selection_summary["selected_split_counts"][split])
            rows.append(
                {
                    "task_family": task_family,
                    "split": split,
                    "family_count": family_count,
                    "candidate_count": family_count * 2,
                    "logical_variant_count": family_count * 4,
                    "invalid_family_count": invalid_family_count,
                    "discarded_family_count": discarded_family_count,
                }
            )
    return rows


def build_full_datasets_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    task_families = config.get("task_families")
    if task_families != ["qa_key", "biorubric"]:
        raise FullBuildError(
            "M5 full_build config must list task_families as ['qa_key', 'biorubric']."
        )

    targets = _validate_targets(config.get("targets", {}))
    qa_fetch_config = _load_nested_config(config, "qakey_fetch_config")
    qa_canonical_config = _load_nested_config(config, "qakey_canonical_config")
    br_fetch_config = _load_nested_config(config, "biorubric_fetch_config")
    br_canonical_config = _load_nested_config(config, "biorubric_canonical_config")

    qakey_summary = _build_qakey_full_freeze(
        fetch_config=qa_fetch_config,
        canonical_config=qa_canonical_config,
        targets=targets,
    )
    biorubric_summary = _build_biorubric_full_freeze(
        fetch_config=br_fetch_config,
        canonical_config=br_canonical_config,
        targets=targets,
    )

    dataset_stats_rows = _build_dataset_stats_rows([qakey_summary, biorubric_summary])
    dataset_stats_path = config.get("dataset_stats_output_path")
    summary_output_path = config.get("summary_output_path")
    if not isinstance(dataset_stats_path, str) or not dataset_stats_path:
        raise FullBuildError("full_build config must contain dataset_stats_output_path.")
    if not isinstance(summary_output_path, str) or not summary_output_path:
        raise FullBuildError("full_build config must contain summary_output_path.")

    _write_csv(
        dataset_stats_path,
        dataset_stats_rows,
        [
            "task_family",
            "split",
            "family_count",
            "candidate_count",
            "logical_variant_count",
            "invalid_family_count",
            "discarded_family_count",
        ],
    )

    summary = {
        "stage": "m5_full_dataset_freeze",
        "targets": dict(targets),
        "task_summaries": {
            "qa_key": qakey_summary,
            "biorubric": biorubric_summary,
        },
        "dataset_stats_output_path": dataset_stats_path,
    }
    write_json(summary_output_path, summary)
    return {
        "summary_output_path": summary_output_path,
        "dataset_stats_output_path": dataset_stats_path,
        "targets": dict(targets),
        "task_summaries": summary["task_summaries"],
    }
