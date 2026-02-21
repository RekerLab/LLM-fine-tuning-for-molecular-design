"""Shared utilities for llmmd: env setup, SMILES processing, model loading."""

import os
from rdkit import Chem
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def setup_env(tmp=None, hf_home=None):
    """Configure TMPDIR and HF_HOME environment variables."""
    if tmp is not None:
        os.environ["TMPDIR"] = tmp
    if hf_home is not None:
        os.environ["HF_HOME"] = hf_home


def process_smiles(smiles):
    """Validate and canonicalize a SMILES string.

    Returns the canonical SMILES or None if invalid.
    """
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def extract_smiles(decoded_output):
    """Extract SMILES from a decoded model output string.

    Looks for the assistant response between Llama-3 chat template markers.
    """
    return (
        decoded_output
        .split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        .split("<|eot_id|>")[0]
        .strip()
    )


def load_inference_model(model_dir, max_seq_length=256):
    """Load a model and tokenizer for inference.

    Returns (model, tokenizer) ready for generation.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

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
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
