from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schema_validation import (  # noqa: E402
    SchemaValidationError,
    validate_entity_fact,
    validate_family_record,
    validate_jsonl_records,
    validate_qa_item,
)
from src.data.toy_records import (  # noqa: E402
    TOY_BIORUBRIC_FAMILY,
    TOY_ENTITY_FACT,
    TOY_QA_FAMILY,
    TOY_QA_ITEM,
)


def _is_family_dataset_path(path: Path) -> bool:
    name = path.name
    return name.endswith("_families.jsonl") and "invalid" not in name and "discarded" not in name


def run_toy_validation() -> None:
    validate_qa_item(TOY_QA_ITEM)
    validate_entity_fact(TOY_ENTITY_FACT)
    validate_family_record(TOY_QA_FAMILY)
    validate_family_record(TOY_BIORUBRIC_FAMILY)
    print("Toy schema validation passed: 2 family records, 1 QA item, 1 entity-fact row.")


def run_full_validation() -> None:
    family_files = sorted(
        path
        for path in Path("data/processed").rglob("*famil*.jsonl")
        if _is_family_dataset_path(path)
    )
    if not family_files:
        raise SchemaValidationError(
            "data/processed: no family JSONL files matching '*famil*.jsonl' were found"
        )

    total_records = 0
    for file_path in family_files:
        record_count = validate_jsonl_records(file_path, validate_family_record)
        print(f"Validated {record_count} family records from {file_path}.")
        total_records += record_count

    print(
        f"Full family schema validation passed across {len(family_files)} files and "
        f"{total_records} records."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate locked v1 schema contracts.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--toy", action="store_true", help="Validate built-in toy examples.")
    mode.add_argument(
        "--full",
        action="store_true",
        help="Validate processed family JSONL files under data/processed.",
    )
    args = parser.parse_args()

    try:
        if args.toy:
            run_toy_validation()
        else:
            run_full_validation()
    except SchemaValidationError as error:
        print(f"Schema validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
