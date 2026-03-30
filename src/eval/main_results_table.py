from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.common.files import atomic_write_text

STANDARD_RUN_GROUP = "m6_standard"
STRICT_RUN_GROUP = "m7_strict"


def _load_slice_payload(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bootstrap = json.loads(summary_path.with_name("bootstrap_ci.json").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "bootstrap": bootstrap,
    }


def _discover_group_summaries(metrics_root: str | Path) -> list[dict[str, Any]]:
    metrics_dir = Path(metrics_root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(metrics_dir.glob("*/summary_metrics.json")):
        rows.append(_load_slice_payload(summary_path))
    if not rows:
        raise ValueError(f"No summary artifacts were found under {metrics_dir}.")
    return rows


def _table_rows(payloads: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        summary = payload["summary"]
        bootstrap = payload["bootstrap"]["metrics"]
        metrics = summary["summary_metrics"]
        rows.append(
            {
                "run_id": summary["run_id"],
                "task_family": summary["task_family"],
                "model_id": summary["model_id"],
                "prompt_id": summary["prompt_id"],
                "split": summary["split"],
                "n_evaluated_families": summary["n_evaluated_families"],
                "base_acc": metrics["base_acc"],
                "base_acc_ci_lower": bootstrap["base_acc"]["ci_lower"],
                "base_acc_ci_upper": bootstrap["base_acc"]["ci_upper"],
                "gcf": metrics["gcf"],
                "gcf_ci_lower": bootstrap["gcf"]["ci_lower"],
                "gcf_ci_upper": bootstrap["gcf"]["ci_upper"],
                "gap": metrics["gap"],
                "gap_ci_lower": bootstrap["gap"]["ci_lower"],
                "gap_ci_upper": bootstrap["gap"]["ci_upper"],
                "sensitivity_at_base_correct": metrics["sensitivity_at_base_correct"],
                "sensitivity_ci_lower": bootstrap["sensitivity_at_base_correct"]["ci_lower"],
                "sensitivity_ci_upper": bootstrap["sensitivity_at_base_correct"]["ci_upper"],
                "invariance_at_base_correct": metrics["invariance_at_base_correct"],
                "invariance_ci_lower": bootstrap["invariance_at_base_correct"]["ci_lower"],
                "invariance_ci_upper": bootstrap["invariance_at_base_correct"]["ci_upper"],
                "order_disagreement": metrics["order_disagreement"],
                "tie_rate": metrics["tie_rate"],
            }
        )
    return rows


def _write_table_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_table_markdown(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _write_baseacc_vs_gcf_figure(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.75", linewidth=1)
    color_by_task = {"qa_key": "#15616d", "biorubric": "#c44536"}
    marker_by_model = {"Llama-3.1-8B-Instruct": "o", "Qwen2.5-14B-Instruct": "s"}

    for row in rows:
        ax.scatter(
            float(row["base_acc"]),
            float(row["gcf"]),
            color=color_by_task[str(row["task_family"])],
            marker=marker_by_model[str(row["model_id"])],
            s=80,
        )
        ax.annotate(
            f"{row['task_family']} / {row['model_id'].replace('-Instruct', '')}",
            (float(row["base_acc"]), float(row["gcf"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Base Accuracy")
    ax.set_ylabel("GCF")
    ax.set_title("M6 Standard Runs: Base Accuracy vs GCF")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_standard_only_artifacts(
    *,
    metrics_root: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Build the first M6 combined table/figure bundle from completed standard slices."""

    payloads = _discover_group_summaries(metrics_root)
    rows = _table_rows(payloads)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "main_results_table.csv"
    markdown_path = output_root / "main_results_table.md"
    figure_path = output_root / "baseacc_vs_gcf.png"

    _write_table_csv(csv_path, rows)
    _write_table_markdown(markdown_path, rows)
    _write_baseacc_vs_gcf_figure(figure_path, rows)

    return {
        "main_results_csv": str(csv_path),
        "main_results_markdown": str(markdown_path),
        "baseacc_vs_gcf_figure": str(figure_path),
    }


def _comparison_key(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    summary = payload["summary"]
    return (
        str(summary["task_family"]),
        str(summary["model_id"]),
        str(summary["split"]),
    )


def _strict_slice_gate(payload: Mapping[str, Any]) -> bool:
    summary = payload["summary"]
    bootstrap = payload["bootstrap"]["metrics"]
    metrics = summary["summary_metrics"]
    gap_gate = float(metrics["gap"]) >= 0.05 and float(bootstrap["gap"]["ci_lower"]) > 0.0
    bio_sensitivity_gate = (
        str(summary["task_family"]) == "biorubric"
        and float(metrics["sensitivity_at_base_correct"]) <= 0.90
        and float(bootstrap["sensitivity_at_base_correct"]["ci_upper"]) < 0.95
    )
    return gap_gate or bio_sensitivity_gate


def _strict_delta_rows(
    standard_payloads: list[Mapping[str, Any]],
    strict_payloads: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    standard_by_key = {_comparison_key(payload): payload for payload in standard_payloads}
    strict_by_key = {_comparison_key(payload): payload for payload in strict_payloads}
    if standard_by_key.keys() != strict_by_key.keys():
        raise ValueError(
            "Strict-delta analysis requires matching standard and strict slice coverage."
        )

    rows: list[dict[str, Any]] = []
    metric_names = (
        "base_acc",
        "gcf",
        "gap",
        "sensitivity_at_base_correct",
        "invariance_at_base_correct",
        "order_disagreement",
        "tie_rate",
    )
    for key in sorted(standard_by_key):
        standard_payload = standard_by_key[key]
        strict_payload = strict_by_key[key]
        standard_summary = standard_payload["summary"]
        strict_summary = strict_payload["summary"]
        strict_bootstrap = strict_payload["bootstrap"]["metrics"]
        standard_metrics = standard_summary["summary_metrics"]
        strict_metrics = strict_summary["summary_metrics"]

        row: dict[str, Any] = {
            "task_family": standard_summary["task_family"],
            "model_id": standard_summary["model_id"],
            "split": standard_summary["split"],
            "n_evaluated_families": standard_summary["n_evaluated_families"],
            "standard_run_id": standard_summary["run_id"],
            "strict_run_id": strict_summary["run_id"],
            "strict_gap_ci_lower": strict_bootstrap["gap"]["ci_lower"],
            "strict_gap_ci_upper": strict_bootstrap["gap"]["ci_upper"],
            "strict_sensitivity_ci_lower": strict_bootstrap["sensitivity_at_base_correct"][
                "ci_lower"
            ],
            "strict_sensitivity_ci_upper": strict_bootstrap["sensitivity_at_base_correct"][
                "ci_upper"
            ],
            "c5_slice_gate_passed": _strict_slice_gate(strict_payload),
        }

        for metric_name in metric_names:
            standard_value = float(standard_metrics[metric_name])
            strict_value = float(strict_metrics[metric_name])
            row[f"{metric_name}_standard"] = standard_value
            row[f"{metric_name}_strict"] = strict_value
            row[f"{metric_name}_delta"] = strict_value - standard_value

        row["strict_gap_gate"] = (
            float(strict_metrics["gap"]) >= 0.05
            and float(strict_bootstrap["gap"]["ci_lower"]) > 0.0
        )
        row["strict_biorubric_sensitivity_gate"] = (
            str(strict_summary["task_family"]) == "biorubric"
            and float(strict_metrics["sensitivity_at_base_correct"]) <= 0.90
            and float(strict_bootstrap["sensitivity_at_base_correct"]["ci_upper"]) < 0.95
        )
        rows.append(row)

    return rows


def _c5_decision_payload(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    supported_rows = [row for row in rows if bool(row["c5_slice_gate_passed"])]
    materially_helped_rows = [row for row in rows if float(row["gap_delta"]) <= -0.01]
    residual_gap_rows = [row for row in rows if float(row["gap_strict"]) > 0.0]

    if supported_rows:
        status = "supported"
        rationale = (
            "At least one strict-prompt slice still clears the locked C5 gate "
            "after strict prompting."
        )
    elif materially_helped_rows and residual_gap_rows:
        status = "partially supported"
        rationale = (
            "Strict prompting reduces the gap on at least one slice, but "
            "positive residual gap remains."
        )
    else:
        status = "unsupported"
        rationale = (
            "Strict prompting does not leave a locked-gate residual gap in the completed slices."
        )

    return {
        "claim_id": "C5",
        "status": status,
        "supported_slice_count": len(supported_rows),
        "materially_helped_slice_count": len(materially_helped_rows),
        "residual_gap_slice_count": len(residual_gap_rows),
        "rationale": rationale,
    }


def _write_gap_delta_figure(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x_positions = [0, 1]
    x_labels = ["standard", "strict"]
    color_by_task = {"qa_key": "#15616d", "biorubric": "#c44536"}
    marker_by_model = {"Llama-3.1-8B-Instruct": "o", "Qwen2.5-14B-Instruct": "s"}

    for row in rows:
        y_values = [float(row["gap_standard"]), float(row["gap_strict"])]
        color = color_by_task[str(row["task_family"])]
        marker = marker_by_model[str(row["model_id"])]
        ax.plot(x_positions, y_values, color=color, marker=marker, linewidth=1.5)
        ax.annotate(
            f"{row['task_family']} / {row['model_id'].replace('-Instruct', '')}",
            (x_positions[-1], y_values[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=8,
        )

    ax.set_xticks(x_positions, x_labels)
    ax.set_ylabel("Gap (BaseAcc - GCF)")
    ax.set_title("M7 Strict vs M6 Standard: Gap by Slice")
    ax.axhline(0.0, linestyle="--", color="0.7", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_c5_decision_note(
    path: str | Path,
    *,
    rows: list[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    lines = [
        "# C5 decision note",
        "",
        f"- Claim status: `{decision['status']}`",
        f"- Rationale: {decision['rationale']}",
        "",
        "## Strict-prompt comparison summary",
    ]
    for row in rows:
        lines.append(
            "- "
            + f"{row['task_family']} / {row['model_id']}: "
            + f"Gap {row['gap_standard']:.6f} -> {row['gap_strict']:.6f} "
            + f"(delta {row['gap_delta']:+.6f}); "
            + f"Sens@BC {row['sensitivity_at_base_correct_standard']:.6f} -> "
            + f"{row['sensitivity_at_base_correct_strict']:.6f}; "
            + f"C5 slice gate passed = {row['c5_slice_gate_passed']}"
        )
    lines.extend(
        [
            "",
            "## Mechanical gate check",
            "- Supported if at least one strict slice still has "
            + "`Gap >= 0.05` with CI lower bound `> 0`, or BioRubric strict "
            + "`Sens@BC <= 0.90` with CI upper bound `< 0.95`.",
            f"- Supported-slice count: {decision['supported_slice_count']}",
            f"- Materially-helped-slice count: {decision['materially_helped_slice_count']}",
            f"- Residual-gap-slice count: {decision['residual_gap_slice_count']}",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def build_strict_delta_artifacts(
    *,
    metrics_root: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Build the M7 strict-vs-standard comparison bundle and C5 decision note."""

    metrics_root_path = Path(metrics_root)
    standard_payloads = _discover_group_summaries(metrics_root_path / STANDARD_RUN_GROUP)
    strict_payloads = _discover_group_summaries(metrics_root_path / STRICT_RUN_GROUP)
    rows = _strict_delta_rows(standard_payloads, strict_payloads)
    decision = _c5_decision_payload(rows)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "strict_delta_table.csv"
    markdown_path = output_root / "strict_delta_table.md"
    figure_path = output_root / "gap_delta.png"
    note_path = output_root / "c5_decision_note.md"
    decision_json_path = output_root / "c5_decision_note.json"

    _write_table_csv(csv_path, rows)
    _write_table_markdown(markdown_path, rows)
    _write_gap_delta_figure(figure_path, rows)
    _write_c5_decision_note(note_path, rows=rows, decision=decision)
    decision_payload = {
        **decision,
        "rows": rows,
    }
    decision_json_path.write_text(
        json.dumps(decision_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return {
        "strict_delta_csv": str(csv_path),
        "strict_delta_markdown": str(markdown_path),
        "gap_delta_figure": str(figure_path),
        "c5_decision_note": str(note_path),
        "c5_decision_json": str(decision_json_path),
    }
