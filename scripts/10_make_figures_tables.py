from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.eval.main_results_table import (  # noqa: E402
    build_standard_only_artifacts,
    build_strict_delta_artifacts,
)
from src.eval.mechanism_diagnostics import (  # noqa: E402
    build_mechanism_diagnostics_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build M6-M9 analysis tables and figures.")
    parser.add_argument("--config", required=True, help="Path to the analysis config.")
    args = parser.parse_args()

    try:
        analysis_config = load_config(args.config)
        analysis_id = analysis_config.get("analysis_id")
        if analysis_id == "standard_only":
            metrics_dir = Path(str(analysis_config["metrics_dir"])) / "m6_standard"
            figure_dir = Path(str(analysis_config["figure_dir"])) / "m6_standard"
            result = build_standard_only_artifacts(
                metrics_root=metrics_dir,
                output_dir=figure_dir,
            )
        elif analysis_id == "strict_delta":
            metrics_dir = Path(str(analysis_config["metrics_dir"]))
            figure_dir = Path(str(analysis_config["figure_dir"])) / "m7_strict"
            result = build_strict_delta_artifacts(
                metrics_root=metrics_dir,
                output_dir=figure_dir,
            )
        elif analysis_id == "mechanism_diagnostics":
            metrics_dir = Path(str(analysis_config["metrics_dir"]))
            figure_dir = Path(str(analysis_config["figure_dir"])) / "mechanism_diagnostics"
            result = build_mechanism_diagnostics_artifacts(
                metrics_root=metrics_dir,
                output_dir=figure_dir,
            )
        else:
            raise ValueError(f"Analysis id {analysis_id!r} is not implemented yet.")
    except Exception as error:
        print(f"Figure/table build failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
