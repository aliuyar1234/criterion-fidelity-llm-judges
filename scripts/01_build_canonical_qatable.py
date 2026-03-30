from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.data.canonical_tables import (  # noqa: E402
    CanonicalSourceError,
    build_canonical_qatable,
    derive_sidecar_paths,
    load_and_validate_relation_whitelist,
    load_jsonl_records,
    write_json,
    write_jsonl_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical QA table for M2.")
    parser.add_argument("--config", required=True, help="Path to the QA-Key source config.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
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

        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "split_manifest_path": str(sidecar_paths["split_manifest_path"]),
                    "qc_report_path": str(sidecar_paths["qc_report_path"]),
                    "output_row_count": len(canonical_rows),
                    "split_anchor_count": len(split_manifest),
                    "skip_reasons": qc_report["skip_reasons"],
                },
                indent=2,
            )
        )
    except (CanonicalSourceError, KeyError, ValueError) as error:
        print(f"Canonical QA table build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
