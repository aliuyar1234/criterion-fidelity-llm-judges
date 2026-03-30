from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.common.files import atomic_write_text
from src.eval.metrics import compute_summary_metrics

RUN_GROUPS = ("m6_standard", "m7_strict")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _float(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(value)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        raise ValueError("Cannot compute a mean over an empty collection.")
    return sum(values_list) / len(values_list)


def _load_run_payloads(metrics_root: str | Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    metrics_root_path = Path(metrics_root)
    for run_group in RUN_GROUPS:
        group_root = metrics_root_path / run_group
        if not group_root.exists():
            continue
        for summary_path in sorted(group_root.glob("*/summary_metrics.json")):
            run_dir = summary_path.parent
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            payloads.append(
                {
                    "run_group": run_group,
                    "summary": summary,
                    "family_rows": _read_csv_rows(run_dir / "family_results.csv"),
                    "variant_rows": _read_csv_rows(run_dir / "variant_results.csv"),
                    "prompt_rows": _read_csv_rows(run_dir / "prompt_results.csv"),
                }
            )
    if not payloads:
        raise ValueError(f"No M6/M7 run payloads were found under {metrics_root_path}.")
    return payloads


def _slice_label(summary: Mapping[str, Any], run_group: str) -> str:
    task = "QA-Key" if str(summary["task_family"]) == "qa_key" else "BioRubric"
    model = "Qwen" if "Qwen" in str(summary["model_id"]) else "Llama"
    prompt = "standard" if run_group == "m6_standard" else "strict"
    return f"{task} / {model} / {prompt}"


def _order_stability_rows(payloads: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset_predicates = {
        "all_families": lambda family_row: True,
        "order_stable": lambda family_row: _int(family_row["order_disagreement_count"]) == 0,
        "order_unstable": lambda family_row: _int(family_row["order_disagreement_count"]) > 0,
        "order_stable_no_ties": lambda family_row: (
            _int(family_row["order_disagreement_count"]) == 0 and _int(family_row["tie_count"]) == 0
        ),
    }

    for payload in payloads:
        summary = payload["summary"]
        family_rows = payload["family_rows"]
        total_families = len(family_rows)
        for subset_name, predicate in subset_predicates.items():
            subset_rows = [dict(row) for row in family_rows if predicate(row)]
            if not subset_rows:
                continue
            summary_metrics = compute_summary_metrics(subset_rows)
            rows.append(
                {
                    "run_group": payload["run_group"],
                    "run_id": summary["run_id"],
                    "slice_label": _slice_label(summary, str(payload["run_group"])),
                    "task_family": summary["task_family"],
                    "model_id": summary["model_id"],
                    "prompt_id": summary["prompt_id"],
                    "subset": subset_name,
                    "n_families": len(subset_rows),
                    "share_of_total_families": len(subset_rows) / total_families,
                    "base_acc": summary_metrics["base_acc"],
                    "gcf": summary_metrics["gcf"],
                    "gap": summary_metrics["gap"],
                    "sensitivity_at_base_correct": summary_metrics["sensitivity_at_base_correct"],
                    "invariance_at_base_correct": summary_metrics["invariance_at_base_correct"],
                    "order_disagreement": summary_metrics["order_disagreement"],
                    "tie_rate": summary_metrics["tie_rate"],
                }
            )
    return rows


def _variant_order_rows(payloads: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        summary = payload["summary"]
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in payload["variant_rows"]:
            grouped.setdefault(str(row["variant_id"]), []).append(dict(row))
        for variant_id, variant_rows in grouped.items():
            rows.append(
                {
                    "run_group": payload["run_group"],
                    "run_id": summary["run_id"],
                    "slice_label": _slice_label(summary, str(payload["run_group"])),
                    "task_family": summary["task_family"],
                    "model_id": summary["model_id"],
                    "prompt_id": summary["prompt_id"],
                    "variant_id": variant_id,
                    "n_variants": len(variant_rows),
                    "order_disagreement_rate": _mean(
                        _int(row["order_disagree"]) for row in variant_rows
                    ),
                    "pred_tie_rate": _mean(_int(row["pred_tie"]) for row in variant_rows),
                    "order_tie_any_rate": _mean(
                        int(_int(row["order_tie_ab"]) or _int(row["order_tie_ba"]))
                        for row in variant_rows
                    ),
                    "variant_error_rate": _mean(
                        int(str(row["pred_winner_cid"]) != str(row["gold_winner_cid"]))
                        for row in variant_rows
                    ),
                }
            )
    return rows


def _chosen_underlying_cid(prompt_row: Mapping[str, str]) -> str | None:
    pred_label = str(prompt_row["pred_display_label"])
    if pred_label == "A":
        return str(prompt_row["displayed_cid_a"])
    if pred_label == "B":
        return str(prompt_row["displayed_cid_b"])
    return None


def _position_preference_rows(payloads: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        summary = payload["summary"]
        prompt_rows = payload["prompt_rows"]
        n_prompts = len(prompt_rows)
        first_count = sum(
            int(str(prompt_row["pred_display_label"]) == "A") for prompt_row in prompt_rows
        )
        second_count = sum(
            int(str(prompt_row["pred_display_label"]) == "B") for prompt_row in prompt_rows
        )
        tie_count = sum(
            int(str(prompt_row["pred_display_label"]) == "TIE") for prompt_row in prompt_rows
        )
        c1_count = sum(
            int(_chosen_underlying_cid(prompt_row) == "c1") for prompt_row in prompt_rows
        )
        c2_count = sum(
            int(_chosen_underlying_cid(prompt_row) == "c2") for prompt_row in prompt_rows
        )
        rows.append(
            {
                "run_group": payload["run_group"],
                "run_id": summary["run_id"],
                "slice_label": _slice_label(summary, str(payload["run_group"])),
                "task_family": summary["task_family"],
                "model_id": summary["model_id"],
                "prompt_id": summary["prompt_id"],
                "n_prompt_instances": n_prompts,
                "first_position_rate": _rate(first_count, n_prompts),
                "second_position_rate": _rate(second_count, n_prompts),
                "position_bias_second_minus_first": _rate(second_count - first_count, n_prompts),
                "label_a_rate": _rate(first_count, n_prompts),
                "label_b_rate": _rate(second_count, n_prompts),
                "tie_rate": _rate(tie_count, n_prompts),
                "choose_c1_rate": _rate(c1_count, n_prompts),
                "choose_c2_rate": _rate(c2_count, n_prompts),
                "underlying_bias_c2_minus_c1": _rate(c2_count - c1_count, n_prompts),
            }
        )
    return rows


def _write_gap_subset_figure(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    subset_order = ["all_families", "order_stable", "order_unstable", "order_stable_no_ties"]
    subset_labels = {
        "all_families": "All",
        "order_stable": "Stable",
        "order_unstable": "Unstable",
        "order_stable_no_ties": "Stable/No Ties",
    }
    slice_order = sorted({str(row["slice_label"]) for row in rows})
    gap_by_key = {(str(row["slice_label"]), str(row["subset"])): _float(row["gap"]) for row in rows}

    values = np.full((len(slice_order), len(subset_order)), np.nan)
    for slice_index, slice_label in enumerate(slice_order):
        for subset_index, subset_name in enumerate(subset_order):
            key = (slice_label, subset_name)
            if key in gap_by_key:
                values[slice_index, subset_index] = gap_by_key[key]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(subset_order)), [subset_labels[name] for name in subset_order])
    ax.set_yticks(range(len(slice_order)), slice_order)
    ax.set_title("Gap by Order-Stability Subset")
    ax.set_xlabel("Subset")
    ax.set_ylabel("Slice")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                ax.text(col_index, row_index, "NA", ha="center", va="center", fontsize=8)
            else:
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="Gap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_variant_order_figure(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    slice_order = sorted({str(row["slice_label"]) for row in rows})
    variant_order = sorted({str(row["variant_id"]) for row in rows})
    disagreement_by_key = {
        (str(row["slice_label"]), str(row["variant_id"])): _float(row["order_disagreement_rate"])
        for row in rows
    }

    values = np.full((len(slice_order), len(variant_order)), np.nan)
    for row_index, slice_label in enumerate(slice_order):
        for col_index, variant_id in enumerate(variant_order):
            key = (slice_label, variant_id)
            if key in disagreement_by_key:
                values[row_index, col_index] = disagreement_by_key[key]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(variant_order)), variant_order)
    ax.set_yticks(range(len(slice_order)), slice_order)
    ax.set_title("Order Disagreement Rate by Variant Type")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Slice")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                ax.text(col_index, row_index, "NA", ha="center", va="center", fontsize=8)
            else:
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="Order Disagreement Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_position_preference_figure(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slice_order = [str(row["slice_label"]) for row in rows]
    first_values = np.array([_float(row["first_position_rate"]) for row in rows])
    second_values = np.array([_float(row["second_position_rate"]) for row in rows])
    tie_values = np.array([_float(row["tie_rate"]) for row in rows])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y_positions = np.arange(len(slice_order))
    ax.barh(y_positions, first_values, color="#15616d", label="First / Label A")
    ax.barh(
        y_positions,
        second_values,
        left=first_values,
        color="#c44536",
        label="Second / Label B",
    )
    ax.barh(
        y_positions,
        tie_values,
        left=first_values + second_values,
        color="#b0b7bf",
        label="Tie",
    )
    ax.set_yticks(y_positions, slice_order)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of prompt instances")
    ax.set_title("Display-Position / Label Preference by Slice")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_diagnostics_note(
    path: str | Path,
    *,
    order_rows: list[Mapping[str, Any]],
    variant_rows: list[Mapping[str, Any]],
    preference_rows: list[Mapping[str, Any]],
) -> None:
    rows_by_slice_subset = {
        (str(row["slice_label"]), str(row["subset"])): row for row in order_rows
    }
    slice_labels = sorted({str(row["slice_label"]) for row in order_rows})
    lines = [
        "# Mechanism Diagnostics Note",
        "",
        "These diagnostics are post-M7 paper-analysis artifacts built from the "
        "saved M6/M7 metric CSVs.",
        "",
        "## 1. Gap by order stability",
    ]
    for slice_label in slice_labels:
        all_row = rows_by_slice_subset.get((slice_label, "all_families"))
        stable_row = rows_by_slice_subset.get((slice_label, "order_stable"))
        unstable_row = rows_by_slice_subset.get((slice_label, "order_unstable"))
        if all_row is None:
            continue
        line = f"- {slice_label}: all-family gap = {float(all_row['gap']):.3f}"
        if stable_row is not None:
            line += (
                f"; order-stable gap = {float(stable_row['gap']):.3f}"
                f" over n={stable_row['n_families']}"
            )
        if unstable_row is not None:
            line += (
                f"; order-unstable gap = {float(unstable_row['gap']):.3f}"
                f" over n={unstable_row['n_families']}"
            )
        lines.append(line)

    lines.extend(["", "## 2. Variant-type order disagreement"])
    highest_variant_rows = sorted(
        variant_rows,
        key=lambda row: float(row["order_disagreement_rate"]),
        reverse=True,
    )[:6]
    for row in highest_variant_rows:
        lines.append(
            "- "
            + f"{row['slice_label']} / {row['variant_id']}: "
            + f"order disagreement = {float(row['order_disagreement_rate']):.3f}, "
            + f"tie rate = {float(row['pred_tie_rate']):.3f}, "
            + f"variant error rate = {float(row['variant_error_rate']):.3f}"
        )

    lines.extend(
        [
            "",
            "## 3. Position / label preference",
            "- In this prompt format, label A corresponds to the "
            "first-presented candidate and label B to the second-presented "
            "candidate.",
        ]
    )
    highest_second_bias_rows = sorted(
        preference_rows,
        key=lambda row: float(row["position_bias_second_minus_first"]),
        reverse=True,
    )
    for row in highest_second_bias_rows[:6]:
        lines.append(
            "- "
            + f"{row['slice_label']}: "
            + f"second-minus-first = {float(row['position_bias_second_minus_first']):+.3f}, "
            + f"tie rate = {float(row['tie_rate']):.3f}, "
            + f"underlying c2-minus-c1 = {float(row['underlying_bias_c2_minus_c1']):+.3f}"
        )

    atomic_write_text(path, "\n".join(lines) + "\n")


def build_mechanism_diagnostics_artifacts(
    *,
    metrics_root: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    payloads = _load_run_payloads(metrics_root)
    order_rows = _order_stability_rows(payloads)
    variant_rows = _variant_order_rows(payloads)
    preference_rows = _position_preference_rows(payloads)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    order_csv = output_root / "order_stability_decomposition.csv"
    order_md = output_root / "order_stability_decomposition.md"
    order_fig = output_root / "order_stability_gap.png"
    variant_csv = output_root / "variant_order_diagnostics.csv"
    variant_md = output_root / "variant_order_diagnostics.md"
    variant_fig = output_root / "variant_order_disagreement.png"
    preference_csv = output_root / "position_label_preference.csv"
    preference_md = output_root / "position_label_preference.md"
    preference_fig = output_root / "position_label_preference.png"
    note_path = output_root / "mechanism_diagnostics_note.md"

    _write_csv(order_csv, order_rows)
    _write_markdown_table(order_md, order_rows)
    _write_gap_subset_figure(order_fig, order_rows)
    _write_csv(variant_csv, variant_rows)
    _write_markdown_table(variant_md, variant_rows)
    _write_variant_order_figure(variant_fig, variant_rows)
    _write_csv(preference_csv, preference_rows)
    _write_markdown_table(preference_md, preference_rows)
    _write_position_preference_figure(preference_fig, preference_rows)
    _write_diagnostics_note(
        note_path,
        order_rows=order_rows,
        variant_rows=variant_rows,
        preference_rows=preference_rows,
    )

    return {
        "order_stability_csv": str(order_csv),
        "order_stability_markdown": str(order_md),
        "order_stability_figure": str(order_fig),
        "variant_diagnostics_csv": str(variant_csv),
        "variant_diagnostics_markdown": str(variant_md),
        "variant_diagnostics_figure": str(variant_fig),
        "position_preference_csv": str(preference_csv),
        "position_preference_markdown": str(preference_md),
        "position_preference_figure": str(preference_fig),
        "mechanism_note": str(note_path),
    }
