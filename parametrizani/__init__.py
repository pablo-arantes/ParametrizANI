"""
ParametrizANI - Fast and Accessible Dihedral Parametrization for Small Molecules
=================================================================================

A production-ready Python package for dihedral parameter optimization using
neural network potentials (TorchANI, AIMNet2, MACE-OFF, GFN2-xTB) and
molecular mechanics (AMBER, OpenFF).

Main Classes
------------
ConformerGenerator : Generate 3D conformers from SMILES/PDB/MOL
ReferenceEnergyCalculator : Calculate energies with ML potentials
EnergyMinimizer : OpenMM energy minimization with restraints
DihedralOptimizer : Optimize dihedral parameters via least-squares fitting
ParameterValidator : Validate optimized parameters with metrics & plots
TopologyGenerator : Generate AMBER/GROMACS/OpenMM topology files

References
----------
Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization
for Small Molecules." J. Chem. Inf. Model. (2025) doi: 10.1021/acs.jcim.5c01957
"""

__version__ = "1.0.0"
__author__ = "Pablo R. Arantes, Souvik Sinha, Giulia Palermo"
__license__ = "MIT"

# Core classes
from .conformer_gen import ConformerGenerator
from .reference_energy import ReferenceEnergyCalculator
from .energy_minimization import EnergyMinimizer
from .dihedral_optimizer import DihedralOptimizer
from .validation import ParameterValidator
from .topology_gen import TopologyGenerator

# Utility functions
from .utils import (
    read_energy_file,
    write_energy_file,
    relative_energies,
    extract_atom_info_from_pdb,
    get_atom_types_from_mol2,
    get_dihedral_atom_types,
)


def generate_conformer(molecule_input, input_type='smiles', work_dir='./work', optimize=True):
    """Quick conformer generation from SMILES/PDB/MOL."""
    gen = ConformerGenerator(molecule_input, input_type, work_dir)
    return gen.run(optimize=optimize)


def calculate_reference_energies(conformer_files, angles=None, method='torchani',
                                  work_dir='./work', optimize=True, device='cpu',
                                  dihedral_indices=None):
    """Quick reference energy calculation for a dihedral scan.
    
    Parameters
    ----------
    method : str
        ML method. Options: 'torchani', 'ani2x', 'ani1x', 'ani1ccx',
        'ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis',
        'wb97m_d3', 'aimnet2', 'b97_3c', 'mace', 'xtb', 'gfn2xtb'
    dihedral_indices : list of int, optional
        Four atom indices defining the dihedral to constrain during optimization.
    """
    calc = ReferenceEnergyCalculator(method, work_dir, device)
    return calc.scan_dihedral(conformer_files, angles, optimize=optimize, fmax=0.001,
                              dihedral_indices=dihedral_indices)


def optimize_dihedral(ref_angles, ref_energies, atom_types=None,
                      mm_energies=None, max_terms=4, work_dir='./work'):
    """Quick dihedral parameter optimization."""
    opt = DihedralOptimizer(max_terms=max_terms, work_dir=work_dir)
    return opt.run_optimization(ref_angles, ref_energies, mm_energies, atom_types)


def validate_parameters(angles, ref_energies, fitted_energies, work_dir='./work'):
    """Quick parameter validation."""
    val = ParameterValidator(work_dir)
    return val.validate_parameters(angles, ref_energies, fitted_energies)


def calculate_resp_charges(mol_file, method='HF', basis='6-31G*', **kwargs):
    """Calculate RESP charges using Psi4 (requires: conda install -c conda-forge psi4 resp).

    Parameters
    ----------
    mol_file : str
        Path to the molecule file (MOL, MOL2, SDF, or PDB format).
    method : str
        QM method for ESP. Options: 'HF', 'B3LYP', 'MP2', 'wB97X-D'.
    basis : str
        Basis set. Options: '6-31G*', 'cc-pVDZ', 'cc-pVTZ'.
    **kwargs
        Additional arguments passed to psi4_utils.calculate_resp_charges().

    Returns
    -------
    dict
        Dictionary with 'charges' (np.ndarray), 'atom_symbols', 'output_file'.
    """
    from .psi4_utils import calculate_resp_charges as _calc_resp
    return _calc_resp(mol_file, method=method, basis=basis, **kwargs)
