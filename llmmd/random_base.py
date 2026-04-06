#!/usr/bin/env python3
# coding=utf-8

"""
Create a randomized base model for control experiments.

Loads Llama-3-8B in full precision, reinitializes all weights randomly,
and saves the result. The saved model can then be loaded with
``load_in_4bit=True`` so that bitsandbytes computes fresh quantization
parameters matched to the random weight distributions.

This proves that pretrained weights are necessary: a model trained from
random initialization should produce nonsensical outputs.

Usage:
    llmmd-random-base --save_dir randomized-llama3 --seed 3407
"""

import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import setup_env


class RandomBaseArgs(Tap):
    model: str = "unsloth/llama-3-8b-Instruct"
    """The base model to randomize (loaded in full precision)."""
    save_dir: str
    """Directory to save the randomized model."""
    seed: int = 3407
    """Random seed for reproducibility."""
    tmp: str = None
    """The temporary directory."""
    HF_HOME: str = None
    """The directory for the HuggingFace model cache."""


def randomize_all_weights(model, seed):
    """Reinitialize all model parameters with random values."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_randomized = 0
    for name, param in model.named_parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param.data)
        elif len(param.shape) == 1:
            if "norm" in name.lower() or "ln" in name.lower():
                if "weight" in name:
                    torch.nn.init.ones_(param.data)
                else:
                    torch.nn.init.zeros_(param.data)
            else:
                torch.nn.init.zeros_(param.data)
        n_randomized += 1

    print(f"Randomized {n_randomized} parameters.")
    return model


def main():
    args = RandomBaseArgs().parse_args()
    setup_env(tmp=args.tmp, hf_home=args.HF_HOME)

    print(f"Loading {args.model} in full precision (this requires ~16GB VRAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Randomizing all weights with seed={args.seed}...")
    model = randomize_all_weights(model, args.seed)

    print(f"Saving randomized model to {args.save_dir}...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # Ensure tokenizer_config.json includes model_type (required by transformers >= 4.57)
    import json, os
    tok_config_path = os.path.join(args.save_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        if "model_type" not in tok_config:
            tok_config["model_type"] = model.config.model_type
            with open(tok_config_path, "w") as f:
                json.dump(tok_config, f, indent=2)

    print("Done. Use this as --model for llmmd-sft training.")


if __name__ == "__main__":
    main()
