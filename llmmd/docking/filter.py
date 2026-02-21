# -*- coding: utf-8 -*-
"""Molecular property calculations: SA score and QED."""

import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")


def calculate_sascore(smiles):
    """Calculate synthetic accessibility score for a SMILES string.

    Returns the SA score (float) or 'failed' on error.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "failed"
        return sascorer.calculateScore(mol)
    except Exception:
        return "failed"


def calculate_qed(smiles):
    """Calculate QED drug-likeness score for a SMILES string.

    Returns the QED score (float) or 'failed' on error.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "failed"
        return QED.qed(mol)
    except Exception:
        return "failed"
