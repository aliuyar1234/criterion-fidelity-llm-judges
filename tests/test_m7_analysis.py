from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.eval.main_results_table import build_strict_delta_artifacts
from src.eval.mechanism_diagnostics import build_mechanism_diagnostics_artifacts


def _write_slice_artifacts(
    root: Path,
    run_group: str,
    run_id: str,
    *,
    task_family: str,
    model_id: str,
    prompt_id: str,
    gap: float,
    gap_ci_lower: float,
    gap_ci_upper: float,
    sensitivity: float,
    sensitivity_ci_lower: float,
    sensitivity_ci_upper: float,
) -> None:
    run_dir = root / run_group / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "run_id": run_id,
        "milestone": "M6" if run_group == "m6_standard" else "M7",
        "task_family": task_family,
        "model_id": model_id,
        "prompt_id": prompt_id,
        "split": "test",
        "dataset_path": f"data/processed/{task_family}/test_families.jsonl",
        "dataset_sha256": "deadbeef",
        "prompt_version": "v1",
        "scoring_version": "continuation_ab_v1",
        "config_fingerprint": "feedface",
        "git_commit": "not_a_git_repo",
        "n_evaluated_families": 600,
        "summary_metrics": {
            "base_acc": 0.80,
            "gcf": 0.80 - gap,
            "gap": gap,
            "sensitivity_at_base_correct": sensitivity,
            "invariance_at_base_correct": 0.95,
            "n_base_correct": 500,
            "n_families": 600,
            "n_logical_variants": 2400,
            "order_disagreement": 0.10,
            "overreaction_at_base_correct": 0.05,
            "underreaction_at_base_correct": 0.10,
            "tie_rate": 0.01,
        },
        "freeze_qc": {
            "invalid_family_count": 0,
            "discarded_family_count": 0,
            "selected_split_counts": {"train": 2000, "dev": 400, "test": 600},
        },
        "run_meta": {
            "state": "COMPLETED",
            "phase": "EXPORT",
            "created_at": "2026-03-30T00:00:00+00:00",
            "updated_at": "2026-03-30T00:00:00+00:00",
        },
    }
    bootstrap_payload = {
        "metrics": {
            "base_acc": {"ci_lower": 0.75, "ci_upper": 0.85},
            "gcf": {"ci_lower": 0.50, "ci_upper": 0.75},
            "gap": {"ci_lower": gap_ci_lower, "ci_upper": gap_ci_upper},
            "sensitivity_at_base_correct": {
                "ci_lower": sensitivity_ci_lower,
                "ci_upper": sensitivity_ci_upper,
            },
            "invariance_at_base_correct": {"ci_lower": 0.93, "ci_upper": 0.97},
        }
    }
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "bootstrap_ci.json").write_text(
        json.dumps(bootstrap_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def test_build_strict_delta_artifacts_writes_c5_note() -> None:
    tmp_path = (Path(".pytest_local_tmp") / "m7_strict_delta").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    metrics_root = tmp_path / "results" / "metrics"
    output_dir = tmp_path / "results" / "figures" / "m7_strict"
    slice_key = "qa_key__qwen__test"

    _write_slice_artifacts(
        metrics_root,
        "m6_standard",
        f"m6__{slice_key}",
        task_family="qa_key",
        model_id="Qwen2.5-14B-Instruct",
        prompt_id="standard",
        gap=0.08,
        gap_ci_lower=0.05,
        gap_ci_upper=0.11,
        sensitivity=0.90,
        sensitivity_ci_lower=0.87,
        sensitivity_ci_upper=0.93,
    )
    _write_slice_artifacts(
        metrics_root,
        "m7_strict",
        f"m7__{slice_key}",
        task_family="qa_key",
        model_id="Qwen2.5-14B-Instruct",
        prompt_id="strict_criterion_emphasis",
        gap=0.06,
        gap_ci_lower=0.03,
        gap_ci_upper=0.09,
        sensitivity=0.92,
        sensitivity_ci_lower=0.90,
        sensitivity_ci_upper=0.94,
    )

    artifacts = build_strict_delta_artifacts(
        metrics_root=metrics_root,
        output_dir=output_dir,
    )

    assert Path(artifacts["strict_delta_csv"]).exists()
    assert Path(artifacts["gap_delta_figure"]).exists()
    note_payload = json.loads(Path(artifacts["c5_decision_json"]).read_text(encoding="utf-8"))
    assert note_payload["status"] == "supported"
    assert note_payload["supported_slice_count"] == 1


def _write_family_variant_prompt_csvs(run_dir: Path) -> None:
    family_csv = "\n".join(
        [
            "family_id,task_family,split,base_correct,paraphrase_all_correct,counterfactual_correct,gcf_success,tie_count,variant_count,order_disagreement_count,attempt_id,completed_at",
            "qa_001,qa_key,test,1,1,1,1,0,4,0,attempt,2026-03-30T00:00:00+00:00",
            "qa_002,qa_key,test,1,1,0,0,1,4,2,attempt,2026-03-30T00:00:00+00:00",
            "",
        ]
    )
    variant_csv = "\n".join(
        [
            "family_id,variant_id,scores_ab_c1,scores_ab_c2,scores_ba_c1,scores_ba_c2,scores_agg_c1,scores_agg_c2,pred_winner_cid,pred_tie,gold_winner_cid,order_pred_ab,order_pred_ba,order_tie_ab,order_tie_ba,order_disagree,attempt_id",
            "qa_001,base,-1,-2,-2,-1,-1.5,-1.5,tie,1,c1,c1,c2,0,0,1,attempt",
            "qa_001,counterfactual,-1,-3,-3,-1,-1,-3,c1,0,c2,c1,c2,0,0,1,attempt",
            "qa_002,base,-1,-3,-3,-1,-1,-3,c1,0,c1,c1,c2,0,0,1,attempt",
            "qa_002,counterfactual,-3,-1,-1,-3,-3,-1,c2,0,c2,c2,c1,0,0,1,attempt",
            "",
        ]
    )
    prompt_csv = "\n".join(
        [
            "family_id,variant_id,order_id,displayed_cid_a,displayed_cid_b,label_text_a,label_text_b,logprob_total_a,logprob_total_b,pred_display_label,score_gap,rendered_prefix_text,rendered_prefix_sha256,rendered_prefix_char_len,attempt_id,inference_ms,label_token_ids_a,label_token_ids_b,label_token_logprobs_a,label_token_logprobs_b",
            'qa_001,base,AB,c1,c2,A,B,-1,-2,A,1,"prefix",hash,6,attempt,1,"[1]","[2]","[-1]","[-2]"',
            'qa_001,base,BA,c2,c1,A,B,-2,-1,B,-1,"prefix",hash,6,attempt,1,"[1]","[2]","[-2]","[-1]"',
            'qa_002,base,AB,c1,c2,A,B,-1,-2,A,1,"prefix",hash,6,attempt,1,"[1]","[2]","[-1]","[-2]"',
            'qa_002,base,BA,c2,c1,A,B,-1,-2,A,1,"prefix",hash,6,attempt,1,"[1]","[2]","[-1]","[-2]"',
            "",
        ]
    )
    (run_dir / "family_results.csv").write_text(family_csv, encoding="utf-8")
    (run_dir / "variant_results.csv").write_text(variant_csv, encoding="utf-8")
    (run_dir / "prompt_results.csv").write_text(prompt_csv, encoding="utf-8")


def test_build_mechanism_diagnostics_artifacts() -> None:
    tmp_path = (Path(".pytest_local_tmp") / "mechanism_diagnostics").resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    metrics_root = tmp_path / "results" / "metrics"
    output_dir = tmp_path / "results" / "figures" / "mechanism_diagnostics"

    for run_group, run_id, prompt_id in (
        ("m6_standard", "m6__qa_key__demo", "standard"),
        ("m7_strict", "m7__qa_key__demo", "strict_criterion_emphasis"),
    ):
        run_dir = metrics_root / run_group / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "run_id": run_id,
            "milestone": "M6" if run_group == "m6_standard" else "M7",
            "task_family": "qa_key",
            "model_id": "Qwen2.5-14B-Instruct",
            "prompt_id": prompt_id,
            "split": "test",
            "n_evaluated_families": 2,
            "summary_metrics": {
                "base_acc": 1.0,
                "gcf": 0.5,
                "gap": 0.5,
                "sensitivity_at_base_correct": 0.5,
                "invariance_at_base_correct": 1.0,
                "order_disagreement": 0.25,
                "tie_rate": 0.125,
            },
        }
        (run_dir / "summary_metrics.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        _write_family_variant_prompt_csvs(run_dir)

    artifacts = build_mechanism_diagnostics_artifacts(
        metrics_root=metrics_root,
        output_dir=output_dir,
    )

    assert Path(artifacts["order_stability_csv"]).exists()
    assert Path(artifacts["variant_diagnostics_csv"]).exists()
    assert Path(artifacts["position_preference_csv"]).exists()
    note_text = Path(artifacts["mechanism_note"]).read_text(encoding="utf-8")
    assert "Gap by order stability" in note_text
    order_csv_text = Path(artifacts["order_stability_csv"]).read_text(encoding="utf-8")
    assert "order_stable" in order_csv_text
