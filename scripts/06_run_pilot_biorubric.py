from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.inference.pilot_runner import PilotRunError, run_and_save_pilot_slice  # noqa: E402
from src.inference.score_ab import (  # noqa: E402
    PRIMARY_MODEL_REPO_IDS,
    TransformersContinuationScorer,
)


def _load_model_config(model_config_path: str) -> dict[str, object]:
    model_config = load_config(model_config_path)
    model_id = model_config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise PilotRunError("Model config must contain a non-empty model_id string.")
    if model_id not in PRIMARY_MODEL_REPO_IDS:
        raise PilotRunError(f"Unsupported model_id {model_id!r} for v1 BioRubric pilot.")
    return model_config


def _load_prompt_id(prompt_config_path: str) -> str:
    prompt_config = load_config(prompt_config_path)
    prompt_id = prompt_config.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise PilotRunError("Prompt config must contain a non-empty prompt_id string.")
    return prompt_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BioRubric pilot inference for one model/prompt."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment config.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require the model/tokenizer to already exist in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--max_families",
        type=int,
        default=None,
        help="Optional cap for engineering-only runs.",
    )
    args = parser.parse_args()

    try:
        exp_config = load_config(args.config)
        dataset_path = exp_config.get("dataset_path")
        model_config_path = exp_config.get("model_config")
        prompt_config_path = exp_config.get("prompt_config")
        split = exp_config.get("split")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise PilotRunError("Experiment config must contain a non-empty dataset_path string.")
        if not isinstance(model_config_path, str) or not model_config_path:
            raise PilotRunError("Experiment config must contain a non-empty model_config string.")
        if not isinstance(prompt_config_path, str) or not prompt_config_path:
            raise PilotRunError("Experiment config must contain a non-empty prompt_config string.")
        if split not in {"train", "dev", "test"}:
            raise PilotRunError("Experiment config must contain split in {train, dev, test}.")

        model_config = _load_model_config(model_config_path)
        model_id = str(model_config["model_id"])
        prompt_id = _load_prompt_id(prompt_config_path)
        repo_id = PRIMARY_MODEL_REPO_IDS[model_id]
        load_in_4bit = bool(model_config.get("load_in_4bit", False))

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            local_files_only=args.local_files_only,
            use_fast=True,
        )
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            local_files_only=args.local_files_only,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quantization_config,
        )
        scorer = TransformersContinuationScorer(model=model, tokenizer=tokenizer)

        result = run_and_save_pilot_slice(
            task_family="biorubric",
            dataset_path=dataset_path,
            model_id=model_id,
            prompt_id=prompt_id,
            split=split,
            tokenizer=tokenizer,
            scorer=scorer,
            max_families=args.max_families,
        )
    except Exception as error:
        print(f"BioRubric pilot inference failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
