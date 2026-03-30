from __future__ import annotations

import re
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_component(value: str) -> str:
    """Convert a config/model identifier into a stable filesystem-safe slug."""

    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Artifact naming components must not be empty after slugification.")
    return slug


def make_run_stem(task_family: str, model_id: str, prompt_id: str, split: str) -> str:
    """Return the canonical artifact stem for a single evaluation slice."""

    return "__".join(
        slugify_component(component) for component in (task_family, model_id, prompt_id, split)
    )


def raw_output_path(task_family: str, model_id: str, prompt_id: str, split: str) -> Path:
    """Return the default raw-output path for a run."""

    return (
        Path("results/raw_outputs")
        / f"{make_run_stem(task_family, model_id, prompt_id, split)}.jsonl"
    )


def metrics_path(task_family: str, model_id: str, prompt_id: str, split: str) -> Path:
    """Return the default metrics artifact path for a run."""

    return (
        Path("results/metrics") / f"{make_run_stem(task_family, model_id, prompt_id, split)}.json"
    )


def token_id_audit_path(model_id: str, prompt_id: str = "standard") -> Path:
    """Return the default token-ID audit path for one model/prompt slice."""

    stem = "__".join((slugify_component(model_id), slugify_component(prompt_id)))
    return Path("results/raw_outputs") / f"token_id_audit__{stem}.json"
