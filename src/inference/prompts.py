from __future__ import annotations

from typing import Any

PROMPT_SPECS: dict[str, dict[str, str]] = {
    "standard": {
        "system": (
            "You are a careful evaluator. Choose which candidate better satisfies "
            "the provided criterion. Output exactly one label."
        ),
    },
    "strict_criterion_emphasis": {
        "system": (
            "You are a criterion-conditioned evaluator. Judge ONLY by the provided "
            "criterion. Do not use outside knowledge, beliefs, or typical preferences "
            "except insofar as they appear in the criterion. If the criterion conflicts "
            "with what you believe, follow the criterion. Output exactly one label."
        ),
    },
}


def build_user_message(
    task_text: str,
    criterion_text: str,
    candidate_a_text: str,
    candidate_b_text: str,
) -> str:
    """Render the locked user message that must end immediately before the label."""

    return "\n\n".join(
        (
            "Task:\n" + task_text,
            "Criterion:\n" + criterion_text,
            "Candidate A:\n" + candidate_a_text,
            "Candidate B:\n" + candidate_b_text,
            "Which candidate better satisfies the criterion?\nFinal label:",
        )
    )


def build_messages(
    prompt_name: str,
    task_text: str,
    criterion_text: str,
    candidate_a_text: str,
    candidate_b_text: str,
) -> list[dict[str, str]]:
    """Build the system/user chat messages for one order-specific prompt."""

    try:
        prompt_spec = PROMPT_SPECS[prompt_name]
    except KeyError as error:
        raise ValueError(f"Unknown prompt name: {prompt_name}") from error

    return [
        {"role": "system", "content": prompt_spec["system"]},
        {
            "role": "user",
            "content": build_user_message(
                task_text=task_text,
                criterion_text=criterion_text,
                candidate_a_text=candidate_a_text,
                candidate_b_text=candidate_b_text,
            ),
        },
    ]


def render_chat_prefix(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render the chat prefix with an assistant generation prompt."""

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered_messages = []
    for message in messages:
        rendered_messages.append(f"{message['role'].upper()}:\n{message['content']}")
    rendered_messages.append("ASSISTANT:\n")
    return "\n\n".join(rendered_messages)
