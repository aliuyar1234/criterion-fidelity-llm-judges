from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.data.wikidata_biorubric_source import (  # noqa: E402
    WikidataBioRubricSourceError,
    fetch_biorubric_source_rows_from_config,
    write_biorubric_source_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch raw BioRubric truthy rows from Wikidata.")
    parser.add_argument("--config", required=True, help="Path to the fetch config.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        rows, summary = fetch_biorubric_source_rows_from_config(
            config,
            progress_callback=lambda payload: print(json.dumps(dict(payload)), flush=True),
        )
        write_biorubric_source_artifacts(
            output_path=config["output_path"],
            summary_path=config["summary_path"],
            rows=rows,
            summary=summary,
        )
    except (KeyError, ValueError, WikidataBioRubricSourceError) as error:
        print(f"BioRubric Wikidata fetch failed: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_path": config["output_path"],
                "summary_path": config["summary_path"],
                "row_count": summary["row_count"],
                "entity_count": summary["entity_count"],
                "slot_row_counts": summary["slot_row_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
