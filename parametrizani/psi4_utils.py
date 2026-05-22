"""
ParametrizANI - Psi4 Utilities
================================

Optional Psi4-based functions for RESP charge calculation.

Requires:
    conda install -c conda-forge psi4 resp

These functions are optional and not required for the main ParametrizANI workflow.
They provide higher-accuracy QM-based charges as an alternative to the default
AM1-BCC charges from antechamber.
"""

import os
import logging
import numpy as np
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def calculate_resp_charges(
    mol_file: str,
    method: str = 'HF',
    basis: str = '6-31G*',
    charge: int = 0,
    multiplicity: int = 1,
    memory: str = '2 GB',
    n_threads: int = 2,
    resp_a: float = 0.0005,
    resp_b: float = 0.1,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate RESP charges using Psi4.

    Uses Psi4 to compute the electrostatic potential (ESP) and fits
    restrained electrostatic potential (RESP) charges. This provides
    higher-accuracy partial charges compared to AM1-BCC.

    Parameters
    ----------
    mol_file : str
        Path to the molecule file (MOL, MOL2, SDF, or PDB format).
    method : str
        QM method for ESP calculation. Options: 'HF', 'B3LYP', 'MP2',
        'wB97X-D'. Default: 'HF'.
    basis : str
        Basis set. Options: '6-31G*', 'cc-pVDZ', 'cc-pVTZ', etc.
        Default: '6-31G*'.
    charge : int
        Net molecular charge. Default: 0.
    multiplicity : int
        Spin multiplicity. Default: 1 (singlet).
    memory : str
        Memory allocation for Psi4. Default: '2 GB'.
    n_threads : int
        Number of CPU threads for Psi4. Default: 2.
    resp_a : float
        RESP restraint strength parameter a. Default: 0.0005.
    resp_b : float
        RESP tightness parameter b. Default: 0.1.
    output_file : str, optional
        Path to save charges. If None, saves to same directory as mol_file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'charges': np.ndarray of RESP charges
        - 'atom_symbols': list of atom element symbols
        - 'output_file': path to saved charges file

    Raises
    ------
    ImportError
        If psi4 or resp packages are not installed.
    ValueError
        If molecule file cannot be read.

    Examples
    --------
    >>> from parametrizani import calculate_resp_charges
    >>> result = calculate_resp_charges('molecule.mol', method='HF', basis='6-31G*')
    >>> print(result['charges'])
    >>> print(f"Total charge: {sum(result['charges']):.4f}")
    """
    try:
        import psi4
    except ImportError:
        raise ImportError(
            "Psi4 is required for RESP charge calculation. "
            "Install with: conda install -c conda-forge psi4"
        )
    try:
        import resp as resp_module
    except ImportError:
        raise ImportError(
            "The 'resp' package is required for RESP charge fitting. "
            "Install with: conda install -c conda-forge resp"
        )

    from rdkit import Chem

    # Read molecule
    mol = None
    ext = os.path.splitext(mol_file)[1].lower()
    if ext in ('.mol', '.sdf'):
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    elif ext == '.mol2':
        mol = Chem.MolFromMol2File(mol_file, removeHs=False)
    elif ext == '.pdb':
        mol = Chem.MolFromPDBFile(mol_file, removeHs=False)

    if mol is None:
        raise ValueError(
            f"Could not read molecule from: {mol_file}. "
            "Supported formats: .mol, .sdf, .mol2, .pdb"
        )

    # Extract atomic coordinates and symbols
    num_atoms = mol.GetNumAtoms()
    atom_symbols = []
    xyz_lines = []

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = mol.GetConformer().GetAtomPosition(i)
        symbol = atom.GetSymbol()
        atom_symbols.append(symbol)
        xyz_lines.append(f"{symbol} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    xyz_string = "\n".join(xyz_lines)

    # Configure Psi4
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)
    psi4.core.set_output_file("psi4_resp.log", False)

    # Build Psi4 molecule
    mol_psi4 = psi4.geometry(f"""
{charge} {multiplicity}
{xyz_string}
""")

    # RESP options
    options = {
        'BASIS_ESP': basis,
        'METHOD_ESP': method,
        'RESP_A': resp_a,
        'RESP_B': resp_b,
    }

    logger.info(f"Running Psi4 RESP: method={method}, basis={basis}, "
                f"charge={charge}, mult={multiplicity}")

    # Calculate RESP charges
    # The resp package API varies between versions:
    #   - Newer (conda-forge): resp.resp() returns (stage1_charges, stage2_charges)
    #     where each is a numpy array of shape (n_atoms,).
    #     Final RESP charges = stage2 = charges_result[1]
    #   - Older (some Colab installs): returns [(grid, charges_array)]
    #     where charges_result[0][1] = charges array
    charges_result = resp_module.resp([mol_psi4], options)
    charges = _extract_resp_charges(charges_result, num_atoms)

    # Save charges
    if output_file is None:
        output_dir = os.path.dirname(mol_file) or '.'
        output_file = os.path.join(output_dir, 'resp_charges.dat')

    np.savetxt(output_file, charges, fmt='%12.6f',
               header=f"RESP charges ({method}/{basis}) for {os.path.basename(mol_file)}")

    logger.info(f"RESP charges calculated for {num_atoms} atoms. "
                f"Total charge: {charges.sum():.4f}")

    return {
        'charges': charges,
        'atom_symbols': atom_symbols,
        'num_atoms': num_atoms,
        'method': method,
        'basis': basis,
        'output_file': output_file,
    }


def _extract_resp_charges(charges_result, num_atoms: int) -> np.ndarray:
    """
    Extract RESP charges from the resp package return value.

    Handles different API versions of the resp package:
    - Newer (conda-forge, Alenaizan): returns (stage1_ndarray, stage2_ndarray)
      where each has shape (n_atoms,). We want stage2 (final RESP).
    - Older versions: may return [(grid_data, charges_array)] where
      charges_result[0][1] is the charges array.

    Parameters
    ----------
    charges_result : tuple or list
        Raw return value from resp.resp().
    num_atoms : int
        Expected number of atoms (for validation).

    Returns
    -------
    np.ndarray
        1D array of RESP charges with shape (num_atoms,).
    """
    # Strategy: try each known API format and validate against num_atoms

    # Attempt 1: Newer API — charges_result[1] is the stage-2 (final) RESP array
    try:
        candidate = np.asarray(charges_result[1], dtype=float).flatten()
        if len(candidate) == num_atoms:
            return candidate
    except (IndexError, TypeError, ValueError):
        pass

    # Attempt 2: Newer API — charges_result[0] is stage-1 (if only 1 stage was run)
    try:
        candidate = np.asarray(charges_result[0], dtype=float).flatten()
        if len(candidate) == num_atoms:
            return candidate
    except (IndexError, TypeError, ValueError):
        pass

    # Attempt 3: Older API — charges_result[0][1] is the charges array
    try:
        candidate = np.asarray(charges_result[0][1], dtype=float).flatten()
        if len(candidate) == num_atoms:
            return candidate
    except (IndexError, TypeError, ValueError):
        pass

    # Attempt 4: Maybe it's a dict with 'charges' key (some wrappers)
    try:
        if hasattr(charges_result, 'get'):
            candidate = np.asarray(charges_result.get('charges', []), dtype=float).flatten()
            if len(candidate) == num_atoms:
                return candidate
    except (TypeError, ValueError):
        pass

    # If nothing worked, provide a helpful error with debug info
    raise RuntimeError(
        f"Could not extract {num_atoms} RESP charges from resp.resp() output. "
        f"Return type: {type(charges_result)}, length: {len(charges_result) if hasattr(charges_result, '__len__') else 'N/A'}. "
        f"Element types: {[type(x).__name__ for x in charges_result] if hasattr(charges_result, '__iter__') else 'not iterable'}. "
        f"Please check your resp package version (conda install -c conda-forge resp)."
    )
