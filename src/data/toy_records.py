from __future__ import annotations

TOY_QA_ITEM = {
    "qa_id": "qa_demo_0001",
    "question": "What is the capital of Austria?",
    "subject_id": "Q40",
    "subject_text": "Austria",
    "relation_id": "P36",
    "relation_name": "capital",
    "answer_id": "Q1741",
    "answer_text": "Vienna",
    "answer_norm": "vienna",
    "answer_aliases_norm": ["wien"],
    "coarse_type": "location",
    "source_ref": "wikidata:Q40:P36",
}

TOY_ENTITY_FACT = {
    "entity_id": "Q42",
    "entity_name": "Douglas Adams",
    "slot": "notable_work",
    "value_id": "Q3107329",
    "value_text": "The Hitchhiker's Guide to the Galaxy",
    "value_norm": "the hitchhiker's guide to the galaxy",
    "source_ref": "wikidata_truthy:Q42:P800",
}

TOY_QA_FAMILY = {
    "family_id": "qakey_demo_0001",
    "task_family": "qa_key",
    "split": "dev",
    "task_text": "What is the capital of Austria?",
    "candidates": [
        {"cid": "c1", "text": "The answer is Vienna."},
        {"cid": "c2", "text": "The answer is Graz."},
    ],
    "variants": [
        {
            "variant_id": "base",
            "kind": "base",
            "criterion_text": "Official answer key: Vienna.",
            "semantics_id": "s_base",
            "gold_winner_cid": "c1",
            "gold_scores": None,
            "metadata": {"template_id": "qa_base"},
        },
        {
            "variant_id": "para_1",
            "kind": "paraphrase",
            "criterion_text": "Grade using this reference answer: Vienna.",
            "semantics_id": "s_base",
            "gold_winner_cid": "c1",
            "gold_scores": None,
            "metadata": {"template_id": "qa_para_1"},
        },
        {
            "variant_id": "para_2",
            "kind": "paraphrase",
            "criterion_text": "Use the following answer key when judging: Vienna.",
            "semantics_id": "s_base",
            "gold_winner_cid": "c1",
            "gold_scores": None,
            "metadata": {"template_id": "qa_para_2"},
        },
        {
            "variant_id": "counterfactual",
            "kind": "counterfactual",
            "criterion_text": "Official answer key: Graz.",
            "semantics_id": "s_counterfactual",
            "gold_winner_cid": "c2",
            "gold_scores": None,
            "metadata": {"template_id": "qa_counterfactual"},
        },
    ],
    "metadata": {
        "source_row_ids": ["qa_demo_0001", "qa_demo_donor_0001"],
        "qc_passed": True,
        "selected_donor_checks": {"is_admissible": True},
    },
}

TOY_BIORUBRIC_FAMILY = {
    "family_id": "biorubric_demo_0001",
    "task_family": "biorubric",
    "split": "test",
    "task_text": "Write a two-sentence biography of Ada Lovelace.",
    "candidates": [
        {
            "cid": "c1",
            "text": (
                "Ada Lovelace was a mathematician. "
                "Ada Lovelace is known for Notes on the Analytical Engine."
            ),
        },
        {
            "cid": "c2",
            "text": (
                "Ada Lovelace was a mathematician. "
                "Ada Lovelace is known for Sketch of the Analytical Engine."
            ),
        },
    ],
    "variants": [
        {
            "variant_id": "base",
            "kind": "base",
            "criterion_text": (
                "Award 1 point(s) if the biography states that Ada Lovelace was a mathematician.\n"
                "Award 2 point(s) if the biography mentions Notes on the Analytical Engine.\n"
                "Award 1 point(s) if the biography mentions Sketch of the Analytical Engine."
            ),
            "semantics_id": "s_base",
            "gold_winner_cid": "c1",
            "gold_scores": {"shared": 1, "f1": 2, "f2": 1},
            "metadata": {"wording_set": 0, "rubric_order": ["shared", "f1", "f2"]},
        },
        {
            "variant_id": "para_lex",
            "kind": "paraphrase",
            "criterion_text": (
                "Give 1 point(s) for stating that Ada Lovelace was a mathematician.\n"
                "Give 2 point(s) for referencing Notes on the Analytical Engine.\n"
                "Give 1 point(s) for referencing Sketch of the Analytical Engine."
            ),
            "semantics_id": "s_base",
            "gold_winner_cid": "c1",
            "gold_scores": {"shared": 1, "f1": 2, "f2": 1},
            "metadata": {"wording_set": 1, "rubric_order": ["shared", "f1", "f2"]},
        },
        {
            "variant_id": "para_struct",
            "kind": "paraphrase",
            "criterion_text": (
                "Award 2 point(s) if the biography mentions Notes on the Analytical Engine.\n"
                "Award 1 point(s) if the biography states that Ada Lovelace was a mathematician.\n"
                "Award 1 point(s) if the biography mentions Sketch of the Analytical Engine."
            ),
            "semantics_id": "s_base",
            "gold_winner_cid": "c1",
            "gold_scores": {"shared": 1, "f1": 2, "f2": 1},
            "metadata": {"wording_set": 0, "rubric_order": ["f1", "shared", "f2"]},
        },
        {
            "variant_id": "counterfactual",
            "kind": "counterfactual",
            "criterion_text": (
                "Award 1 point(s) if the biography states that Ada Lovelace was a mathematician.\n"
                "Award 1 point(s) if the biography mentions Notes on the Analytical Engine.\n"
                "Award 2 point(s) if the biography mentions Sketch of the Analytical Engine."
            ),
            "semantics_id": "s_counterfactual",
            "gold_winner_cid": "c2",
            "gold_scores": {"shared": 1, "f1": 1, "f2": 2},
            "metadata": {"wording_set": 0, "rubric_order": ["shared", "f1", "f2"]},
        },
    ],
    "metadata": {
        "source_row_ids": ["Q7259"],
        "qc_passed": True,
        "candidate_checks": {
            "both_candidates_factual_by_source": True,
            "shared_sentence_identical": True,
            "same_distinguishing_template": True,
            "c1_exactly_two_sentences": True,
            "c2_exactly_two_sentences": True,
            "no_extra_facts": True,
            "candidate_length_ratio_in_range": True,
        },
    },
}
