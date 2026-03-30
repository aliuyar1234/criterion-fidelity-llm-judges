from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.data.biorubric_builder import build_biorubric_from_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build locked BioRubric family artifacts.")
    parser.add_argument("--config", required=True, help="Path to the BioRubric build config.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = build_biorubric_from_config(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
