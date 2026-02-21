# LLM Fine-Tuning for Molecular Design

Fine-tuning Large Language Models (Llama-3-8B) via Supervised Fine-Tuning (SFT) to generate drug-like molecules with desired binding affinities to target proteins.

## Overview

This repository provides tools for:
1. **Supervised Fine-Tuning (SFT)** of Llama-3-8B (4-bit quantized via Unsloth) with LoRA adapters
2. **Molecule Generation** from fine-tuned models at fixed or ramping temperatures
3. **Docking Evaluation** of generated molecules against target proteins using QuickVina/Smina

## Installation

```bash
# Create conda environment
mamba create -n llm-mol python=3.10
mamba activate llm-mol

# Install dependencies
mamba install openbabel rdkit -c conda-forge
pip install -r requirements.txt

# Install tartarus (docking package)
pip install -e .
```

## Usage

### 1. Supervised Fine-Tuning

```bash
python sft.py \
    --data_path data_so_1syh_sft.json \
    --save_dir so-1syh-160-32-32-128 \
    --epochs 160 \
    --save_epochs_stride 16 \
    --rank 32 \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 128
```

For control experiments with random weight initialization:
```bash
python sft.py ... --random_weights
```

### 2. Molecule Generation (Fixed Temperature)

```bash
python molecule_generation.py \
    --save_dir output_dir \
    --model_dir path/to/checkpoint \
    --temperature 0.6 \
    --n_mols 100 \
    --prompt "Generate SMILES of a molecule with strong binding affinity to protein 1syh"
```

### 3. Molecule Generation (Temperature Ramping)

```bash
python efficient_molecule_generation.py \
    --save_dir output_dir \
    --model_dir path/to/checkpoint \
    --temperature_start 0.1 \
    --dt 0.1 \
    --n_mols 10000 \
    --stop_cutoff 0.2 \
    --data_path path/to/training_smiles.csv \
    --prompt "Generate SMILES of a molecule with strong binding affinity to protein 1syh"
```

### 4. Docking Evaluation

```bash
python docking_evaluation.py input_molecules.csv protein_name
```

Where `protein_name` is one of: `1syh`, `4lde`, `6y2f`.

## Example Training Data

Example training data is provided in the `examples/` directory:

### Docking Tasks (`examples/docking/`)

6 training datasets (10,000 samples each) for 3 proteins x 2 objectives:

| File | Protein | Objective |
|------|---------|-----------|
| `data_so_1syh_sft.json` | 1SYH | Single-objective (binding affinity) |
| `data_so_4lde_sft.json` | 4LDE | Single-objective (binding affinity) |
| `data_so_6y2f_sft.json` | 6Y2F | Single-objective (binding affinity) |
| `data_mo_1syh_sft.json` | 1SYH | Multi-objective (binding + drug-likeness + synthesizability) |
| `data_mo_4lde_sft.json` | 4LDE | Multi-objective (binding + drug-likeness + synthesizability) |
| `data_mo_6y2f_sft.json` | 6Y2F | Multi-objective (binding + drug-likeness + synthesizability) |

### Excipient-Nanoparticle Task (`examples/excipient-nanoparticle/`)

| File | Samples | Description |
|------|---------|-------------|
| `data_sft.json` | 1,522 | Excipient generation for drug-nanoparticle stabilization |

### Data Format

Training data is a JSON file with `prompt` and `response` keys:
```json
{
    "prompt": ["Generate SMILES of a molecule with strong binding affinity to protein 1syh", ...],
    "response": ["CCO...", ...]
}
```

## Acknowledgments

Docking evaluation is based on [Tartarus](https://github.com/aspuru-guzik-group/Tartarus).
