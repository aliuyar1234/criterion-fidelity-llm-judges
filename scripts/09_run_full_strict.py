from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.full_runner import (  # noqa: E402
    FullBaselineRunError,
    group_slice_specs_by_model,
    load_model_bundle,
    release_model_bundle,
    resolve_slice_specs,
    run_standard_slice,
    write_matrix_status,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M7 full strict-prompt baselines.")
    parser.add_argument(
        "--config", required=True, help="Path to the full-strict experiment config."
    )
    parser.add_argument("--family-type", default=None, help="Optional task-family filter.")
    parser.add_argument("--model-id", default=None, help="Optional model-id filter.")
    parser.add_argument("--prompt-id", default=None, help="Optional prompt-id filter.")
    parser.add_argument("--split", default=None, help="Optional split override.")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Accepted for compatibility; matching slices run by default.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for compatibility; matching slices auto-resume when safe.",
    )
    parser.add_argument(
        "--only-postprocess",
        action="store_true",
        help="Skip inference and regenerate exports/bootstrap/audits from existing SQLite ledgers.",
    )
    parser.add_argument(
        "--stop-after-families",
        type=int,
        default=None,
        help="Gracefully pause after this many newly completed families in each slice.",
    )
    parser.add_argument(
        "--force-takeover",
        action="store_true",
        help="Replace an existing lock file for the matching slice if necessary.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the model/tokenizer to already exist in the local Hugging Face cache.",
    )
    args = parser.parse_args()

    try:
        slice_specs = resolve_slice_specs(
            exp_config_path=args.config,
            family_type=args.family_type,
            model_id=args.model_id,
            prompt_id=args.prompt_id,
            split_override=args.split,
        )
        if not slice_specs:
            raise FullBaselineRunError("No full-strict slices matched the requested filters.")

        write_matrix_status(slice_specs)
        results: list[dict[str, object]] = []
        grouped_specs = group_slice_specs_by_model(slice_specs)

        for model_config_path, model_slice_specs in grouped_specs.items():
            model_bundle = None
            try:
                if not args.only_postprocess:
                    model_bundle = load_model_bundle(
                        model_config_path,
                        local_files_only=args.local_files_only,
                    )

                for spec in model_slice_specs:
                    result = run_standard_slice(
                        spec=spec,
                        tokenizer=None if model_bundle is None else model_bundle["tokenizer"],
                        scorer=None if model_bundle is None else model_bundle["scorer"],
                        only_postprocess=args.only_postprocess,
                        stop_after_families=args.stop_after_families,
                        force_takeover=args.force_takeover,
                    )
                    results.append(result)
                    write_matrix_status(slice_specs)
            finally:
                if model_bundle is not None:
                    release_model_bundle(model_bundle)

    except Exception as error:
        print(f"M7 full strict baseline run failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps({"slice_results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
