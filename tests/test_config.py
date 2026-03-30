from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from src.common.artifacts import make_run_stem, metrics_path, raw_output_path
from src.common.config import ConfigError, load_config


def test_loads_json_compatible_yaml_config() -> None:
    config = load_config("configs/data/qakey_source.yaml")

    assert config["task_family"] == "qa_key"
    assert len(config["relation_whitelist"]) == 6


def test_rejects_non_mapping_config() -> None:
    bad_config = "tests/_bad_config_for_validation.yaml"
    try:
        with open(bad_config, "w", encoding="utf-8") as handle:
            handle.write('["not", "a", "mapping"]')

        with pytest.raises(ConfigError):
            load_config(bad_config)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(bad_config).unlink()


def test_artifact_naming_convention() -> None:
    stem = make_run_stem("qa_key", "Qwen2.5-14B-Instruct", "strict criterion-emphasis", "test")

    assert stem == "qa-key__qwen2-5-14b-instruct__strict-criterion-emphasis__test"
    assert (
        raw_output_path("qa_key", "Qwen2.5-14B-Instruct", "standard", "dev")
        .as_posix()
        .endswith("results/raw_outputs/qa-key__qwen2-5-14b-instruct__standard__dev.jsonl")
    )
    assert (
        metrics_path("biorubric", "Llama-3.1-8B-Instruct", "standard", "test")
        .as_posix()
        .endswith("results/metrics/biorubric__llama-3-1-8b-instruct__standard__test.json")
    )
