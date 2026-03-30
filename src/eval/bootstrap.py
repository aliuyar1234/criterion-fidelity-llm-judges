from __future__ import annotations

import random
from typing import Any, Mapping

from src.eval.metrics import compute_summary_metrics


def bootstrap_summary_metrics(
    family_results: list[Mapping[str, Any]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compute 95% family-bootstrap confidence intervals for the headline metrics."""

    if not family_results:
        raise ValueError("Cannot bootstrap an empty family-results slice.")

    metric_names = (
        "base_acc",
        "gcf",
        "gap",
        "sensitivity_at_base_correct",
        "invariance_at_base_correct",
        "underreaction_at_base_correct",
        "overreaction_at_base_correct",
        "tie_rate",
        "order_disagreement",
    )
    point_estimates = compute_summary_metrics(family_results)
    samples_by_metric: dict[str, list[float]] = {metric_name: [] for metric_name in metric_names}
    rng = random.Random(seed)
    family_results_list = [dict(row) for row in family_results]

    for _ in range(int(n_bootstrap)):
        resampled = [
            family_results_list[rng.randrange(len(family_results_list))]
            for _ in range(len(family_results_list))
        ]
        summary = compute_summary_metrics(resampled)
        for metric_name in metric_names:
            value = summary.get(metric_name)
            if value is None:
                continue
            samples_by_metric[metric_name].append(float(value))

    ci_by_metric: dict[str, Any] = {}
    for metric_name in metric_names:
        values = sorted(samples_by_metric[metric_name])
        if not values:
            ci_by_metric[metric_name] = {
                "point_estimate": point_estimates.get(metric_name),
                "ci_lower": None,
                "ci_upper": None,
            }
            continue

        lower_index = max(0, int(0.025 * len(values)) - 1)
        upper_index = min(len(values) - 1, int(0.975 * len(values)))
        ci_by_metric[metric_name] = {
            "point_estimate": point_estimates.get(metric_name),
            "ci_lower": values[lower_index],
            "ci_upper": values[upper_index],
        }

    return {
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "n_families": len(family_results_list),
        "metrics": ci_by_metric,
    }
