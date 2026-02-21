#!/usr/bin/env python3
# coding=utf-8

"""
Docking evaluation CLI.

Reads an input CSV with a 'smiles' column, computes docking scores, SA scores,
and QED for each unique SMILES, and writes results to *_output.csv.

Usage:
    llmmd-dock --input_csv input.csv --protein 1syh
    llmmd-dock --input_csv input.csv --protein 4lde --bin /path/to/obabel/bin/ --num_cores 4
"""

import multiprocessing
import os
import shutil

import pandas as pd
from tap import Tap
from tqdm import tqdm

from .core import perform_calc_single
from .filter import calculate_qed, calculate_sascore


class DockArgs(Tap):
    input_csv: str
    """Path to input CSV file with a 'smiles' column."""
    protein: str
    """Protein name for docking: 1syh, 4lde, or 6y2f."""
    bin: str = None
    """Path to directory containing obabel binary. Auto-detected if not set."""
    num_cores: int = 8
    """Number of parallel processes for docking."""
    n_repeats: int = 5
    """Number of docking repeats per molecule (best score kept)."""


def _find_obabel_bin(explicit_bin):
    """Resolve the obabel binary directory.

    Priority: explicit arg > OBABEL_BIN env var > PATH lookup.
    """
    if explicit_bin is not None:
        return explicit_bin
    env_bin = os.environ.get("OBABEL_BIN")
    if env_bin is not None:
        return env_bin
    obabel_path = shutil.which("obabel")
    if obabel_path is not None:
        return os.path.dirname(obabel_path) + "/"
    return ""


def _calculate_docking_score(args_tuple):
    """Worker function for multiprocessing docking."""
    smiles, protein, bin_dir, n_repeats = args_tuple
    try:
        scores = [
            perform_calc_single(smiles, protein, docking_program="qvina", bin=bin_dir, save_pdb=True)
            for _ in range(n_repeats)
        ]
        score = min(scores) if min(scores) < 0.0 else 10.0
        return score
    except Exception:
        return 10.0


def main():
    args = DockArgs().parse_args()
    bin_dir = _find_obabel_bin(args.bin)

    tqdm.pandas()
    df = pd.read_csv(args.input_csv)
    smiles4docking = df.smiles.unique().tolist()
    df1 = pd.DataFrame({"smiles": smiles4docking})
    print(f"Number of unique SMILES: {len(smiles4docking)}")

    df1["sascore"] = df1.smiles.progress_apply(calculate_sascore)
    df1["qed"] = df1.smiles.progress_apply(calculate_qed)

    work_items = [
        (smi, args.protein, bin_dir, args.n_repeats) for smi in smiles4docking
    ]
    with multiprocessing.Pool(args.num_cores) as pool:
        docking_scores = list(
            tqdm(pool.imap(_calculate_docking_score, work_items), total=len(work_items))
        )
    df1[f"{args.protein}_score"] = docking_scores

    output_csv = args.input_csv.replace(".csv", "_output.csv")
    df1.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
