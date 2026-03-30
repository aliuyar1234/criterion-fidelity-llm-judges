from __future__ import annotations

FAMILY_SPLITS = {"train", "dev", "test"}
PRIMARY_TASK_FAMILIES = {"qa_key", "biorubric"}
PRIMARY_CANDIDATE_IDS = {"c1", "c2"}
TIE_EPSILON = 1e-6

QAKEY_RELATION_SPECS: dict[str, dict[str, str]] = {
    "P19": {
        "relation_name": "place_of_birth",
        "coarse_type": "location",
        "question_template": "Where was {SUBJECT} born?",
    },
    "P20": {
        "relation_name": "place_of_death",
        "coarse_type": "location",
        "question_template": "Where did {SUBJECT} die?",
    },
    "P569": {
        "relation_name": "date_of_birth",
        "coarse_type": "date",
        "question_template": "When was {SUBJECT} born?",
    },
    "P570": {
        "relation_name": "date_of_death",
        "coarse_type": "date",
        "question_template": "When did {SUBJECT} die?",
    },
    "P36": {
        "relation_name": "capital",
        "coarse_type": "location",
        "question_template": "What is the capital of {SUBJECT}?",
    },
    "P38": {
        "relation_name": "currency",
        "coarse_type": "currency",
        "question_template": "What currency is used in {SUBJECT}?",
    },
}

BIORUBRIC_ALLOWED_SLOTS = {"occupation", "notable_work", "award", "field"}
BIORUBRIC_SHARED_SLOT = "occupation"
BIORUBRIC_DISTINGUISHING_SLOT_PRIORITY = ["notable_work", "award", "field"]
BIORUBRIC_SLOT_PROPERTY_IDS = {
    "occupation": "P106",
    "notable_work": "P800",
    "award": "P166",
    "field": "P101",
}

TASK_FAMILY_VARIANT_IDS: dict[str, tuple[str, str, str, str]] = {
    "qa_key": ("base", "para_1", "para_2", "counterfactual"),
    "biorubric": ("base", "para_lex", "para_struct", "counterfactual"),
}

TASK_FAMILY_PARAPHRASE_IDS: dict[str, tuple[str, str]] = {
    "qa_key": ("para_1", "para_2"),
    "biorubric": ("para_lex", "para_struct"),
}
