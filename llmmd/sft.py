#!/usr/bin/env python3
# coding=utf-8

"""
Large Language Model Supervised Fine-tuning for Molecular Generation.

Fine-tunes Llama-3-8B (4-bit quantized via Unsloth) using LoRA adapters.
Supports random-weights control experiments and checkpoint resumption.

Usage:
    llmmd-sft --data_path data.json --save_dir output --epochs 160 ...
"""

import json
import math
import os
import re

import pandas as pd
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from tap import Tap
from trl import SFTConfig, SFTTrainer

from .utils import setup_env


class SftArgs(Tap):
    model: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    """The name of the model to fine-tune."""
    data_path: str
    """The path of the training data for fine-tuning."""
    save_dir: str
    """The directory to save the results."""
    epochs: int = 1
    """The number of epochs for fine-tuning."""
    rank: int = 16
    """The rank of the LoRA adapters."""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps."""
    per_device_train_batch_size: int = 128
    """The batch size per device."""
    save_epochs_stride: int = 1
    """The frequency to save the checkpoint model."""
    tmp: str = None
    """The temporary directory to save the intermediate results."""
    HF_HOME: str = None
    """The directory to save the foundation LLM model, such as Llama3."""
    seed: int = 3407
    """Random seed."""
    random_weights: bool = False
    """If True, reinitialize all pretrained weights randomly for control experiment."""


def random_weights_control(model, seed=None):
    """Reinitialize all pretrained parameters with random weights.

    Creates a control model to demonstrate the importance of pretrained weights.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print("=" * 50)
    print("WARNING: Reinitializing all pretrained weights!")
    print("This is a control experiment - expect poor performance.")
    print("=" * 50)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                continue
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

    print("All pretrained weights have been randomly reinitialized.")
    print("=" * 50)

    return model


def check_checkpoint_directories(base_dir):
    """Return True if any checkpoint-N directories exist in base_dir."""
    pattern = re.compile(r"^checkpoint-\d+$")
    entries = os.listdir(base_dir)
    return any(
        os.path.isdir(os.path.join(base_dir, entry)) and pattern.match(entry)
        for entry in entries
    )


def main():
    args = SftArgs().parse_args()
    setup_env(tmp=args.tmp, hf_home=args.HF_HOME)

    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    if args.random_weights:
        model = random_weights_control(model, seed=args.seed)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt",
        },
    )

    def formatting_prompts_func(examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt, response in zip(prompts, responses)
        ]
        return {"text": texts}

    with open(args.data_path, "r") as file:
        data_dict = json.load(file)
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    n_steps_per_epoch = math.ceil(
        len(data_dict["prompt"])
        / args.per_device_train_batch_size
        / args.gradient_accumulation_steps
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.1,
            save_strategy="steps",
            save_steps=args.save_epochs_stride * n_steps_per_epoch,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.save_dir,
        ),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    if check_checkpoint_directories(args.save_dir):
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        trainer_stats = trainer.train()

    pd.DataFrame(trainer.state.log_history[:-2]).to_csv(
        f"{args.save_dir}/training_log.csv", index=False
    )
    model.save_pretrained(f"{args.save_dir}/lora_model")


if __name__ == "__main__":
    main()
