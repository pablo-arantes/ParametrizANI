"""
ParametrizANI - Utility Functions
=================================

Common helper functions used across modules.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Constants
SMALL_REAL = 1.0E-8
KCAL_TO_KJ = 4.184
HARTREE_TO_KCAL = 627.5094740631


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the given name and level."""
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def create_work_dir(work_dir: str) -> str:
    """Create working directory if it doesn't exist."""
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def read_energy_file(file_path: str) -> Tuple[List[float], List[float]]:
    """
    Read an energy profile file with columns: angle energy.
    
    Parameters
    ----------
    file_path : str
        Path to the data file.
        
    Returns
    -------
    Tuple[List[float], List[float]]
        Tuple of (angles, energies).
    """
    angles = []
    energies = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    angles.append(float(parts[0]))
                    energies.append(float(parts[1]))
                except ValueError:
                    continue
    
    return angles, energies


def write_energy_file(file_path: str, angles: List[float], energies: List[float]) -> None:
    """
    Write an energy profile file.
    
    Parameters
    ----------
    file_path : str
        Output file path.
    angles : List[float]
        Dihedral angles in degrees.
    energies : List[float]
        Energy values in kcal/mol.
    """
    with open(file_path, 'w') as f:
        for angle, energy in zip(angles, energies):
            f.write(f"{angle:.3f} {energy:.6f}\n")


def relative_energies(energies: np.ndarray) -> np.ndarray:
    """Convert absolute energies to relative energies (min = 0)."""
    energies = np.asarray(energies, dtype=float)
    return energies - energies.min()


def extract_atom_info_from_pdb(pdb_file_path: str) -> List[Dict[str, Any]]:
    """
    Extract atom information from a PDB file.
    
    Parameters
    ----------
    pdb_file_path : str
        Path to PDB file.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of atom dictionaries with keys: index, name, element, x, y, z.
    """
    atom_info = []
    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_number = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                element = line[76:78].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_info.append({
                    'index': atom_number - 1,
                    'name': atom_name,
                    'element': element,
                    'x': x, 'y': y, 'z': z
                })
    return atom_info


def validate_dihedral_indices(indices: List[int], num_atoms: int) -> None:
    """
    Validate dihedral atom indices.
    
    Parameters
    ----------
    indices : List[int]
        List of 4 atom indices.
    num_atoms : int
        Total number of atoms in the molecule.
        
    Raises
    ------
    ValueError
        If indices are invalid.
    """
    if len(indices) != 4:
        raise ValueError(f"Dihedral requires exactly 4 atom indices, got {len(indices)}")
    
    for idx in indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range [0, {num_atoms-1}]")
    
    if len(set(indices)) != 4:
        raise ValueError("All 4 atom indices must be unique")


def get_atom_types_from_mol2(mol2_file: str) -> List[str]:
    """
    Extract GAFF/SYBYL atom types from a MOL2 file.
    
    Parameters
    ----------
    mol2_file : str
        Path to MOL2 file (e.g., from antechamber).
        
    Returns
    -------
    List[str]
        List of atom type strings, one per atom.
    """
    atom_types = []
    in_atom_block = False
    
    with open(mol2_file, 'r') as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            elif line.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if in_atom_block:
                parts = line.split()
                if len(parts) >= 6:
                    # Column 6 (index 5) is the atom type in MOL2 format
                    atom_types.append(parts[5].lower())
    
    return atom_types


def get_dihedral_atom_types(mol2_file: str, dihedral_indices: List[int]) -> List[str]:
    """
    Get atom types for a specific dihedral from a MOL2 file.
    
    Parameters
    ----------
    mol2_file : str
        Path to MOL2 file with atom type assignments.
    dihedral_indices : List[int]
        Four atom indices defining the dihedral.
        
    Returns
    -------
    List[str]
        Four atom type strings for the dihedral atoms.
    """
    all_types = get_atom_types_from_mol2(mol2_file)
    
    if not all_types:
        logger.warning(f"No atom types found in {mol2_file}")
        return ['X', 'X', 'X', 'X']
    
    dihedral_types = []
    for idx in dihedral_indices:
        if idx < len(all_types):
            dihedral_types.append(all_types[idx])
        else:
            logger.warning(f"Atom index {idx} out of range (max {len(all_types)-1})")
            dihedral_types.append('X')
    
    return dihedral_types
