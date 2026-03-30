from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import ConfigError, load_config  # noqa: E402
from src.data.biorubric_builder import BioRubricBuildError  # noqa: E402
from src.data.canonical_tables import CanonicalSourceError  # noqa: E402
from src.data.full_build import FullBuildError, build_full_datasets_from_config  # noqa: E402
from src.data.qakey_builder import QAKeyBuildError  # noqa: E402
from src.data.wikidata_biorubric_source import WikidataBioRubricSourceError  # noqa: E402
from src.data.wikidata_qakey_source import WikidataQAKeySourceError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and freeze the full M5 datasets.")
    parser.add_argument("--config", required=True, help="Path to the full-build config.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = build_full_datasets_from_config(config)
    except (
        BioRubricBuildError,
        CanonicalSourceError,
        ConfigError,
        FullBuildError,
        KeyError,
        QAKeyBuildError,
        ValueError,
        WikidataBioRubricSourceError,
        WikidataQAKeySourceError,
    ) as error:
        print(f"Full dataset build failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
