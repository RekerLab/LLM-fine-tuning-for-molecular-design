# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llmmd` — a Python package for fine-tuning Llama-3-8B (4-bit quantized via Unsloth) with LoRA adapters to generate SMILES strings of drug molecules with desired binding affinities to target proteins. The package provides three CLI tools: SFT training, molecule generation, and docking evaluation. Based on [Tartarus](https://github.com/aspuru-guzik-group/Tartarus) for docking evaluation.

## Installation

```bash
mamba create -n llm-mol python=3.10
mamba activate llm-mol
mamba install openbabel rdkit -c conda-forge
pip install -e .
```

Note: `openbabel` and `rdkit` must be installed via conda/mamba (not pip). The package itself is installed as editable (`pip install -e .`) which registers the three CLI entry points.

## CLI Entry Points

Defined in `pyproject.toml [project.scripts]`. All use `typed-argument-parser` (Tap) — run any with `--help` for full argument docs.

| Command | Module | Purpose |
|---------|--------|---------|
| `llmmd-sft` | `llmmd.sft:main` | Supervised fine-tuning with LoRA adapters |
| `llmmd-molgen` | `llmmd.molgen:main` | Molecule generation (fixed or ramping temperature) |
| `llmmd-dock` | `llmmd.docking.evaluate:main` | Docking evaluation with QuickVina/Smina |

### SFT Training

```bash
llmmd-sft \
    --data_path examples/docking/data_so_1syh_sft.json \
    --save_dir so-1syh-160-32-32-128 \
    --epochs 160 --save_epochs_stride 16 --rank 32 \
    --gradient_accumulation_steps 32 --per_device_train_batch_size 128 \
    --HF_HOME /work/yx240/hub --tmp /work/yx240/hub/tmp
```

Key flags: `--random_weights` reinitializes all pretrained weights for control experiments. Training auto-resumes from the latest checkpoint if `checkpoint-N/` directories exist in `--save_dir`.

### Molecule Generation

```bash
# Fixed temperature (100 molecules)
llmmd-molgen --mode fixed --save_dir out --model_dir checkpoint-160 \
    --temperature 0.6 --n_mols 100 \
    --prompt "Generate SMILES of a molecule with strong binding affinity to protein 1syh"

# Temperature ramping with early stopping (10k molecules)
llmmd-molgen --mode efficient --save_dir out --model_dir checkpoint-160 \
    --temperature_start 0.1 --dt 0.1 --n_mols 10000 \
    --stop_cutoff 0.2 --data_path training_smiles.csv \
    --prompt "Generate SMILES of a molecule with strong binding affinity to protein 1syh"
```

Both modes skip generation if `generated_mols.csv` already exists in `--save_dir`. Efficient mode saves progress incrementally and can resume.

### Docking Evaluation

```bash
llmmd-dock --input_csv generated_mols.csv --protein 1syh --num_cores 8
```

Supported proteins: `1syh`, `4lde`, `6y2f`. Outputs `*_output.csv` with columns: `smiles`, `sascore`, `qed`, `{protein}_score`. Requires `obabel` on PATH (or set `--bin` / `OBABEL_BIN` env var).

## Architecture

```
llmmd/
├── sft.py           # SFT training (FastLanguageModel + SFTTrainer from trl)
├── molgen.py        # Generation: fixed temp or temperature-ramping with early stopping
├── utils.py         # Shared: env setup, SMILES extraction/validation, model loading
└── docking/
    ├── core.py      # QuickVina/Smina docking via subprocess (bundled binaries in data/)
    ├── filter.py    # SA score and QED calculation via RDKit
    ├── evaluate.py  # CLI wrapper: parallel docking with multiprocessing.Pool
    ├── data/        # Bundled: qvina, smina binaries
    └── structures/  # Protein structures: {1syh,4lde,6y2f}/prot.pdbqt
```

Key design decisions:
- Model loading uses Unsloth's `FastLanguageModel` with Llama-3 chat template throughout (`utils.py:load_inference_model`)
- SMILES extraction parses Llama-3 chat template markers (`<|start_header_id|>assistant<|end_header_id|>` ... `<|eot_id|>`)
- Docking binaries (qvina, smina) are bundled as package data, not system dependencies
- `calculate_sascore` and `calculate_qed` return the string `"failed"` (not NaN) on error — use `pd.to_numeric(col, errors='coerce')` when reading output CSVs
- Docking returns `10**4` (or `10.0`) as sentinel for failed calculations
- Each molecule is docked `n_repeats` times (default 5) and the best score is kept

## Training Data Format

JSON with `prompt` and `response` keys (lists of equal length):
```json
{
    "prompt": ["Generate SMILES of a molecule with strong binding affinity to protein 1syh", ...],
    "response": ["CCO...", ...]
}
```

Example datasets in `examples/docking/` (3 proteins x 2 objectives, 10k samples each) and `examples/excipient-nanoparticle/`.

## Testing

No test suite exists yet. There are no `tests/` directory or test configurations. When adding tests, note that GPU-dependent code (sft, molgen) will need mocking or a GPU-equipped environment.

## Environment

- Runs on Duke HPC (DCC) with SLURM scheduler
- HuggingFace cache: `/work/yx240/hub`
- GPU: RTXA6000 preferred, requires CUDA 7.0+
- Key deps: unsloth, transformers, trl, rdkit, typed-argument-parser, pandas
- `unsloth_compiled_cache/` at repo root is auto-generated by Unsloth — do not edit
