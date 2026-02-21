# -*- coding: utf-8 -*-
"""Core docking functions: QuickVina/Smina docking for 1syh, 4lde, 6y2f.

Ported from tartarus/docking.py with paths updated to use package data.
"""

import os
import subprocess
import time
import uuid

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_PKG_DIR, "data")
_STRUCT_DIR = os.path.join(_PKG_DIR, "structures")


def check_energy(lig_, bin=""):
    """Compute ligand total energy using Open Babel obenergy.

    Returns the total energy in kcal/mol, or 10000 on failure.
    """
    try:
        ob_cmd = [os.path.join(f"{bin}obenergy"), lig_]
        result = subprocess.run(ob_cmd, capture_output=True)
        line = result.stdout.decode("utf-8").split("\n")[-2]
        total_energy = float(line.split(" ")[-2])
    except Exception:
        total_energy = 10000
    return total_energy


def _parse_docking_score(command_run):
    """Extract docking score from qvina/smina stdout."""
    lines = command_run.stdout.decode("utf-8").split("\n")
    docking_score = []
    for item in lines:
        parts = [x for x in item.split(" ") if x != ""]
        if len(parts) == 4:
            try:
                float(parts[0])
                vr_2 = float(parts[1])
                float(parts[2])
                float(parts[3])
                docking_score.append(vr_2)
            except ValueError:
                continue
    return min(docking_score)


def _run_docking(lig_location, out_location, protein, method="qvina", bin=""):
    """Run docking for a given protein using qvina or smina."""
    configs = {
        "1syh": {
            "center": ("21.492800140380858", "13.457733376820881", "23.175899950663247"),
            "size": ("20", "20", "20"),
        },
        "4lde": {
            "center": ("-2.942962976793448", "-12.915592617458767", "-50.99233344749168"),
            "size": ("20", "20", "20"),
        },
        "6y2f": {
            "center": ("11.026168696851615", "-0.6082891440804464", "20.840999947973046"),
            "size": ("10", "10", "10"),
        },
    }

    if protein not in configs:
        raise ValueError(f"Unknown protein: {protein!r}. Supported: {list(configs)}")

    cfg = configs[protein]
    cx, cy, cz = cfg["center"]
    sx, sy, sz = cfg["size"]

    if method == "qvina":
        program = os.path.join(_DATA_DIR, "qvina")
        exhaustiveness = "10"
    elif method == "smina":
        program = os.path.join(_DATA_DIR, "smina")
        exhaustiveness = "100"
    else:
        raise ValueError(f"Possible docking softwares: qvina/smina, got {method!r}")

    receptor = os.path.join(_STRUCT_DIR, protein, "prot.pdbqt")

    command_run = subprocess.run(
        [
            program,
            "--receptor", receptor,
            "--ligand", lig_location,
            "--center_x", cx,
            "--center_y", cy,
            "--center_z", cz,
            "--size_x", sx,
            "--size_y", sy,
            "--size_z", sz,
            "--exhaustiveness", exhaustiveness,
            "--out", out_location,
        ],
        capture_output=True,
    )

    pose_energy = check_energy(out_location, bin=bin)
    if pose_energy == 10000:
        return 10000

    return _parse_docking_score(command_run)


def generate_unique_file_name(base_name, extension):
    """Generate a unique filename with timestamp and UUID."""
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"


def perform_calc_single(smi, receptor_type, docking_program="qvina", bin="", save_pdb=False):
    """Perform docking calculation on a single molecule.

    Parameters
    ----------
    smi : str
        SMILES string of the molecule.
    receptor_type : str
        Protein identifier: '1syh', '4lde', or '6y2f'.
    docking_program : str
        'qvina' or 'smina'.
    bin : str
        Path to directory containing obabel binary.
    save_pdb : bool
        If True, keep intermediate pdbqt files.

    Returns
    -------
    float
        Docking score, or 10**4 on failure.
    """
    output_filename = generate_unique_file_name("lig", "pdbqt")
    out_location = generate_unique_file_name("pose", "pdbqt")
    try:
        cmd = [f"{bin}obabel", "-ismi", "-:" + smi, "-O", output_filename, "--gen3d"]
        subprocess.run(cmd, timeout=60)

        lig_energy = check_energy(output_filename, bin=bin)
        if lig_energy >= 10000:
            return 10**4

        score_ = _run_docking(
            output_filename, out_location, receptor_type,
            method=docking_program, bin=bin,
        )
    except Exception:
        return 10**4
    finally:
        if not save_pdb:
            for f in (output_filename, out_location):
                if os.path.exists(f):
                    os.remove(f)

    return score_
