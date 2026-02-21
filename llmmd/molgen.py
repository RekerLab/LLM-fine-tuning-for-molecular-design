#!/usr/bin/env python3
# coding=utf-8

"""
Molecule generation using fine-tuned language models.

Supports two modes:
  --mode fixed     : Fixed temperature, generates N molecules (was molecule_generation.py)
  --mode efficient : Temperature ramping with early stopping (was efficient_molecule_generation.py)

Usage:
    llmmd-molgen --mode fixed --save_dir out --model_dir checkpoint --temperature 0.6 --n_mols 100 --prompt "..."
    llmmd-molgen --mode efficient --save_dir out --model_dir checkpoint --temperature_start 0.1 --dt 0.1 ...
"""

import os

import pandas as pd
import transformers
from tap import Tap
from tqdm import tqdm

from .utils import extract_smiles, load_inference_model, process_smiles, setup_env


class MolgenArgs(Tap):
    mode: str = "fixed"
    """Generation mode: 'fixed' (constant temperature) or 'efficient' (temperature ramping)."""
    save_dir: str
    """The directory to save the generated molecules."""
    model_dir: str
    """The directory of the fine-tuned model."""
    prompt: str
    """The prompt command to generate molecules."""
    n_mols: int = 100
    """The number of molecules to generate."""
    num_beams: int = 1
    """The number of beams for beam search."""
    # Fixed mode args
    temperature: float = 0.6
    """The temperature for sampling (fixed mode)."""
    # Efficient mode args
    temperature_start: float = 0.1
    """The start temperature for sampling (efficient mode)."""
    dt: float = 0.1
    """The increment of temperature for sampling (efficient mode)."""
    stop_cutoff: float = None
    """If generated/attempts < this value, increase temperature (efficient mode)."""
    data_path: str = None
    """Path to CSV of training SMILES to exclude (efficient mode)."""
    # Environment
    tmp: str = None
    """The temporary directory to save the intermediate results."""
    HF_HOME: str = None
    """The directory to save the foundation LLM model, such as Llama3."""


def _generate_one(model, tokenizer, prompt_text, temperature, num_beams, max_seq_length=256):
    """Generate a single molecule and return the raw SMILES string."""
    messages = [{"from": "human", "value": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_length=max_seq_length,
        temperature=temperature,
        do_sample=True,
        num_beams=num_beams,
    )
    decoded = tokenizer.batch_decode(outputs)
    return extract_smiles(decoded[0])


def _run_fixed(args, model, tokenizer):
    """Fixed-temperature generation: produce exactly n_mols molecules."""
    output_path = os.path.join(args.save_dir, "generated_mols.csv")
    if os.path.exists(output_path):
        return

    prompt_text = f"{args.prompt}."
    smiles_list = []
    valid_list = []

    for _ in tqdm(range(args.n_mols), total=args.n_mols):
        smiles = _generate_one(model, tokenizer, prompt_text, args.temperature, args.num_beams)
        csmiles = process_smiles(smiles)
        if csmiles is None:
            smiles_list.append(smiles)
            valid_list.append(False)
        else:
            smiles_list.append(csmiles)
            valid_list.append(True)

    df = pd.DataFrame({"smiles": smiles_list, "validness": valid_list})
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def _run_efficient(args, model, tokenizer):
    """Temperature-ramping generation with uniqueness filtering and early stopping."""
    output_path = os.path.join(args.save_dir, "generated_mols.csv")

    train_smiles = []
    if args.data_path is not None:
        train_smiles = pd.read_csv(args.data_path).smiles.tolist()

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        smiles_list = df["smiles"].tolist()
        temperature_list = df["temperature"].tolist()
        temperature = temperature_list[-1]
    else:
        smiles_list = []
        temperature_list = []
        temperature = args.temperature_start

    with tqdm(total=args.n_mols, desc="Generating molecules") as pbar:
        pbar.update(len(smiles_list))
        while len(smiles_list) < args.n_mols:
            if temperature > 2.0:
                break
            batch = []
            for _ in range(20):
                smiles = _generate_one(
                    model, tokenizer, args.prompt, temperature, args.num_beams
                )
                csmiles = process_smiles(smiles)
                if (
                    csmiles is not None
                    and csmiles not in train_smiles
                    and csmiles not in smiles_list
                    and csmiles not in batch
                ):
                    batch.append(csmiles)
                    temperature_list.append(temperature)
                    pbar.update(1)

            if args.stop_cutoff is not None and len(batch) / 20 < args.stop_cutoff:
                temperature += args.dt
            smiles_list += batch

            df = pd.DataFrame({"smiles": smiles_list, "temperature": temperature_list})
            os.makedirs(args.save_dir, exist_ok=True)
            df.to_csv(output_path, index=False)


def main():
    args = MolgenArgs().parse_args()
    setup_env(tmp=args.tmp, hf_home=args.HF_HOME)

    transformers.set_seed(0)
    model, tokenizer = load_inference_model(args.model_dir)

    if args.mode == "fixed":
        _run_fixed(args, model, tokenizer)
    elif args.mode == "efficient":
        _run_efficient(args, model, tokenizer)
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}. Use 'fixed' or 'efficient'.")


if __name__ == "__main__":
    main()
