from __future__ import annotations

from copy import deepcopy

import pytest

from src.data.schema_validation import (
    SchemaValidationError,
    validate_entity_fact,
    validate_family_record,
    validate_qa_item,
)
from src.data.toy_records import (
    TOY_BIORUBRIC_FAMILY,
    TOY_ENTITY_FACT,
    TOY_QA_FAMILY,
    TOY_QA_ITEM,
)


def test_toy_records_validate() -> None:
    validate_qa_item(TOY_QA_ITEM)
    validate_entity_fact(TOY_ENTITY_FACT)
    validate_family_record(TOY_QA_FAMILY)
    validate_family_record(TOY_BIORUBRIC_FAMILY)


def test_family_validator_rejects_wrong_variant_set() -> None:
    broken_family = deepcopy(TOY_QA_FAMILY)
    broken_family["variants"][1]["variant_id"] = "para_lex"

    with pytest.raises(SchemaValidationError):
        validate_family_record(broken_family)


def test_family_validator_rejects_unflipped_counterfactual_gold() -> None:
    broken_family = deepcopy(TOY_BIORUBRIC_FAMILY)
    broken_family["variants"][-1]["gold_winner_cid"] = "c1"

    with pytest.raises(SchemaValidationError):
        validate_family_record(broken_family)


def test_qa_item_validator_checks_locked_relation_mapping() -> None:
    broken_item = deepcopy(TOY_QA_ITEM)
    broken_item["coarse_type"] = "date"

    with pytest.raises(SchemaValidationError):
        validate_qa_item(broken_item)
