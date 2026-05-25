# ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules

![alt text](https://github.com/pablo-arantes/ParametrizANI/blob/main/TOC_graphic.png)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Hi there!

Welcome to ParametrizANI, an innovative and free tool designed to address the growing demand for accurate parametrization of small molecules in molecular studies. Our goal is to democratize research by providing a research-friendly environment that is free from resource constraints, enabling teams of all sizes to perform dihedral parametrization with DFT-level accuracy.

We have now integrated **TorchANI2** into our pipeline, along with the latest improved ANI models trained on the expanded 2× dataset (**ANI-2xr, ANI-2dr, ANI-2xr-Snn, and ANI-mbis**).
These models were trained at the B97-3c level of theory and include explicit repulsion and dispersion corrections, smoother potential energy surfaces (PES), and MBIS-derived charges.

ParametrizANI is a production-ready Python package for dihedral parameter optimization using neural network potentials and molecular mechanics. It provides an end-to-end pipeline from SMILES strings to force field parameters compatible with AMBER, GROMACS, and OpenMM.

## Notebooks

**Notebook A** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_GAFF2.ipynb) - `Dihedral parametrization of small molecules for GAFF force field using state-of-the-art reference methods such as TorchANI, AIMNet2, MACE-OFF or GFN2-xTB.`

**Notebook B** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_OpenFF.ipynb) - `Dihedral parametrization of small molecules for OpenFF force fields using state-of-the-art reference methods such as TorchANI, AIMNet2, MACE-OFF or GFN2-xTB.`

**Notebook C** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_TorchANI%2BPsi4.ipynb) - `Dihedral parametrization of small molecules using a reference potential computed with Psi4, combined with structural optimization from TorchANI.`

**Notebook D** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RotProf.ipynb) - `Rotational Profile – fits an empirical energy profile to a reference profile, which can be obtained experimentally, through quantum mechanical (QM) calculations, or using machine learning models such as TorchANI.`

**Notebook E** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RESP_charges.ipynb) - `Perform a geometry optimization with TorchANI, AIMNet2, or GFN2-xTB, then compute RESP charges to generate the GAFF topology parameters.`

## Key Features and Benefits

**• Robust Backbone:** ParametrizANI leverages TorchANI, a robust PyTorch-based deep learning program, as its benchmark to ensure precision in parametrization tasks. TorchANI is crucial for training and inference of ANI (ANAKIN-ME) deep learning models, which are fundamental for predicting potential energy surfaces and other molecular system attributes.

**• Accuracy and Efficiency:** ParametrizANI establishes detailed protocols for dihedral parametrization using both GAFF and OpenFF force fields. By integrating TorchANI's predictive power, ParametrizANI offers a streamlined and accurate approach to parametrization, especially for small molecules. TorchANI's neural network models predict molecular energies and properties with high accuracy and efficiency, significantly reducing computation time compared to traditional Quantum Mechanical (QM) methods.

**• Cloud-Based Accessibility:** The tool harnesses the power of Google Colaboratory (Colab), a hosted Jupyter Notebook service that provides free access to computing resources. This makes ParametrizANI a feasible, cost-effective, and accessible approach to compound parametrization, particularly beneficial for investigators worldwide, including those with limited resources. Our notebooks are designed to run efficiently on CPU cores, requiring no heavy parallel processing.

**• Comprehensive Workflow:** ParametrizANI provides comprehensive workflows implemented in Google Colab notebooks, exemplifying a complete pipeline for dihedral parametrization from SMILES strings generation to force field parameter optimization. These workflows enable researchers to efficiently perform accurate and reliable dihedral parametrization.

**• Versatile and Customizable:** The notebooks are designed for ease of use, following the Jupyter Notebook structure, with an initial configuration step taking less than 5 minutes. Users can select between GAFF and OpenFF force fields, choose charge calculation methods (AM1-BCC or RESP), and even upload their own reference energy profiles calculated using external software (e.g., Gaussian, GAMESS). This flexibility allows for customization to specific research requirements and professional use.

**• Broad Applicability:** ParametrizANI is not only suited for advanced molecular dynamics research and computational drug discovery but also serves as an excellent tool for educational purposes. It allows students to independently run the entire parametrization process without local software compilation or extensive coding experience, with embedded visualization at each step.

## Installation

### Step 1: Install conda dependencies (required)

Several packages (`openmm`, `openff-toolkit`, `ambertools`, `rdkit`, `xtb-python`) are only available via conda-forge:

```bash
conda install -c conda-forge ambertools openmm openmmforcefields rdkit openbabel openff-toolkit xtb-python -y
```

### Step 2: Install parametrizani

```bash
cd parametrizani
pip install -e .
```

### Step 3: Install ML potentials (pip)

```bash
# TorchANI
pip install torch torchani

# MACE-OFF (optional)
pip install mace-torch e3nn==0.4.4

# AIMNet2 (optional) - uses torch.jit models from the cloned repo
git clone https://github.com/pablo-arantes/AIMNet2.git
```

### Quick install (all pip extras at once)

```bash
pip install -e ".[full]"
```

> **Note:** The `[full]` extra installs only pip-available packages (`torchani`, `mace-torch`, `parmed`, `ase`). You still need conda for `openmm`, `openff-toolkit`, `ambertools`, `rdkit`, and `xtb-python`.

### Step 4 (Optional): Psi4 for RESP charges

```bash
conda install -c conda-forge psi4 resp -y
```

Psi4 enables QM-based RESP charge calculation (HF/6-31G*, B3LYP, MP2) as an alternative to AM1-BCC.
It is **not required** for the main workflow.

### Google Colab

On Colab, everything is handled automatically via condacolab. See the provided notebooks.

## Quick Start

```python
from parametrizani import (
    ConformerGenerator, TopologyGenerator, EnergyMinimizer,
    calculate_reference_energies, optimize_dihedral, get_dihedral_atom_types,
)

# Define dihedral indices (0-based atom indices for the rotatable bond)
dihedral_indices = [0, 1, 2, 3]

# 1. Generate a 3D conformer and dihedral scan
gen = ConformerGenerator('CC(=O)OC', 'smiles', './work')
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

# 2. Generate topology & detect atom types automatically from MOL2
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)

# 3. Calculate reference energies (with constrained geometry optimization)
ref = calculate_reference_energies(
    scan['conformers'], scan['angles'], method='torchani',
    dihedral_indices=dihedral_indices
)

# 4. MM minimization with dihedral zeroed (isolates torsion contribution)
minimizer = EnergyMinimizer('gaff2', './work')
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)

# 5. Optimize dihedral parameters (fits ref - mm_zeroed)
opt = optimize_dihedral(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)
print(f"RMSE: {opt['rmse']:.4f} kcal/mol")
print(f"Atom types: {atom_types}")  # e.g. ['c3', 'c', 'o', 'c3']
```

### Full Workflow with Classes (GAFF2)

```python
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
    get_dihedral_atom_types,
)
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the molecule and dihedral
smiles = 'COc3ccc2c(=O)cc(c1ccccc1)oc2c3'
dihedral_indices = [8, 9, 10, 15]
workDir = './work'
model_name = 'torchani'

# 1. Generate conformer and dihedral scan
gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

# 2. Generate initial topology & detect atom types
topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
print(f"Detected atom types: {atom_types}")  # e.g. ['ca', 'ca', 'os', 'ca']

# 3. Calculate reference energies (geometry optimized with dihedral constrained)
calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=0.001,  # Convergence threshold for geometry optimization
    dihedral_indices=dihedral_indices
)

# 4. MM minimization with dihedral zeroed (to isolate torsion contribution)
minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)

# --- Plot: Reference vs MM ---
min_deg, max_deg = min(ref['angles']), max(ref['angles'])
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label="GAFF2")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference vs GAFF2 Energy Profile')
plt.savefig(f'{model_name}_vs_gaff2.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Optimize dihedral parameters
optimizer = DihedralOptimizer(max_terms=6, work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)

# --- Print RMSE and optimized parameters ---
print(f"\nRMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# --- Plot: Optimized Profile ---
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.0, label="GAFF2 (original)", alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label="Optimized")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Validate
validator = ParameterValidator(workDir)
validation = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"Quality: {validation['quality']} (RMSE: {validation['rmse']:.4f} kcal/mol)")

# 7. Write FRCMOD and update topology with optimized parameters
IDIVF = 1       # Scaling factor for equivalent torsions
n_terms = 6     # Number of Fourier terms to use

selected_frcmod_params = optimizer.format_frcmod_params(
    result, n_terms=n_terms, idivf=IDIVF
)
frcmod_file = optimizer.write_frcmod(result, idivf=IDIVF, n_terms=n_terms)

# Use the MOL2 with charges (new.mol2 if available, otherwise ligand.mol2)
topology_mol2 = os.path.join(topo.tleap_dir, 'new.mol2')
if not os.path.exists(topology_mol2):
    topology_mol2 = amber_files['mol2']

# Update FRCMOD: zeroes central-pair dihedrals, removes exact matches, inserts new params
updated_frcmod = topo.update_frcmod(
    amber_files['frcmod'],
    selected_frcmod_params,
    dihedral_indices=dihedral_indices,
    mol2_file=topology_mol2,
)

# 8. Generate final topology with optimized parameters
new_amber = topo.generate_amber(
    conf['mol_file'],
    frcmod_file=updated_frcmod,
    mol2_file=topology_mol2,
    output_prefix='SYS_new',
)
print(f"Updated topology: {new_amber['prmtop']}")

# 9. Export to GROMACS or OpenMM format
gromacs_files = topo.generate_gromacs(new_amber['prmtop'], new_amber['inpcrd'])
openmm_files = topo.generate_openmm(new_amber['prmtop'], new_amber['inpcrd'])

# 10. Verify: re-run MM scan with new topology (zero_dihedral=False to test full potential)
mm_new = minimizer.minimize_scan(
    new_amber['prmtop'], new_amber['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=False
)

# --- Plot: Verification ---
plt.figure(figsize=(10, 6))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=2, markersize=6,
         label=model_name + ' (Reference)')
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.5,
         label='GAFF2 (original, dihedral=0)', alpha=0.6)
plt.plot(result['angles'], result['best_fit'], 'D--', linewidth=1.5,
         label='Fitted curve', alpha=0.7)
plt.plot(mm_new['angles'], mm_new['energies_relative'], '^-', linewidth=2,
         markersize=6, label='GAFF2 (new parameters)', color='green')
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)', fontsize=12)
plt.ylabel('Relative Energy (kcal/mol)', fontsize=12)
plt.legend(frameon=True, fontsize=10)
plt.title('Verification: Optimized Parameters in Final Topology', fontsize=13)
plt.tight_layout()
plt.savefig('verification_final.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print verification RMSE ---
rmse_new = np.sqrt(np.mean(
    (np.array(ref['energies_relative']) - np.array(mm_new['energies_relative']))**2
))
print(f"\nRMSE (Reference vs New Topology): {rmse_new:.4f} kcal/mol")
if rmse_new < 1.0:
    print("  Parameters validated successfully!")
else:
    print("  RMSE > 1.0 - consider adjusting the number of Fourier terms")
```

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `conformer_gen` | `ConformerGenerator` | Generate 3D conformers from SMILES/PDB/MOL |
| `reference_energy` | `ReferenceEnergyCalculator` | ML potential energy calculations |
| `energy_minimization` | `EnergyMinimizer` | OpenMM minimization with restraints |
| `dihedral_optimizer` | `DihedralOptimizer` | Least-squares dihedral fitting |
| `validation` | `ParameterValidator` | Quality metrics and visualization |
| `topology_gen` | `TopologyGenerator` | AMBER/GROMACS/OpenMM topology generation |
| `utils` | `get_dihedral_atom_types` | Auto-detect atom types from MOL2 |
| `psi4_utils` | `calculate_resp_charges` | QM RESP charges via Psi4 (optional) |

## Supported ML Methods

| Method | Package | Level of Theory |
|--------|---------|------------------|
| ANI-2x | TorchANI | ωB97X/6-31G* |
| ANI-1x | TorchANI | ωB97X/6-31G* |
| ANI-1ccx | TorchANI | CCSD(T)/CBS |
| ANI-2xr | TorchANI | B97-3c |
| ANI-2dr | TorchANI | B97-3c |
| ANI-2xr-Snn | TorchANI | B97-3c |
| ANI-mbis | TorchANI | B97-3c |
| MACE-OFF | MACE | DFT (SPICE) |
| GFN2-xTB | xtb-python | GFN2-xTB |
| AIMNet2 (wB97M-D3) | AIMNet2 | ωB97M-D3 |
| AIMNet2 (B97-3c) | AIMNet2 | B97-3c |

### How to select a model

The ML model is selected via the `method` parameter in `ReferenceEnergyCalculator`:

```python
from parametrizani import ReferenceEnergyCalculator

# Use any of the supported method strings:
calc = ReferenceEnergyCalculator('ani2xr', './work')  # TorchANI ANI-2xr
```

**Available `method` values:**

| `method` string | Model | Package required |
|-----------------|-------|------------------|
| `torchani` or `ani2x` | ANI-2x (default) | `torchani` |
| `ani1x` | ANI-1x | `torchani` |
| `ani1ccx` | ANI-1ccx | `torchani` |
| `ani2xr` | ANI-2xr | `torchani` |
| `ani2dr` | ANI-2dr | `torchani` |
| `ani2xr_snn` | ANI-2xr-Snn | `torchani` |
| `ani_mbis` | ANI-mbis | `torchani` |
| `wb97m_d3` or `aimnet2` | AIMNet2 (wB97M-D3) | AIMNet2 (git clone) |
| `b97_3c` | AIMNet2 (B97-3c) | AIMNet2 (git clone) |
| `mace` | MACE-OFF (medium) | `mace-torch` |
| `xtb` or `gfn2xtb` | GFN2-xTB | `xtb-python` (conda) |

In the Colab notebooks, select the model from the dropdown widget in the "Reference Energy Calculation" cell.

### How to select the force field (GAFF2 vs OpenFF)

The force field choice affects two steps: **topology generation** and **MM minimization**.

**GAFF2 workflow** (uses antechamber + tleap):

```python
from parametrizani import TopologyGenerator, EnergyMinimizer

dihedral_indices = [8, 9, 10, 15]

# Topology with GAFF2
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')

# MM minimization with GAFF2 (zeroes ALL torsions around the central bond)
minimizer = EnergyMinimizer('gaff2', './work')
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)
```

**OpenFF workflow** (uses SMIRNOFF force fields):

```python
from parametrizani import EnergyMinimizer

dihedral_indices = [8, 9, 10, 15]

# MM minimization with OpenFF (no topology step needed)
minimizer = EnergyMinimizer('openff-2.0.0', './work')
mm = minimizer.minimize_scan_openff(
    smiles,                  # SMILES string for the molecule
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)
```

**Available force field values for `EnergyMinimizer`:**

| Value | Force Field | Notes |
|-------|-------------|-------|
| `gaff2` | GAFF2 | Requires AmberTools (antechamber, tleap) |
| `openff-2.0.0` | OpenFF 2.0.0 (Sage) | Recommended OpenFF version |
| `openff-1.3.1` | OpenFF 1.3.1 | Legacy |
| `openff-1.2.0` | OpenFF 1.2.0 | Legacy |
| `smirnoff99Frosst-1.1.0` | SMIRNOFF99Frosst | Original SMIRNOFF |

> **Key difference:** GAFF2 requires generating topology files first (`TopologyGenerator.generate_amber()`), while OpenFF works directly from the SMILES string — no separate topology step needed for minimization.

In the Colab notebooks, use **Notebook A** for GAFF2 or **Notebook B** for OpenFF.

### How to select the charge method

The charge method is set via the `charge_method` parameter in `TopologyGenerator.generate_amber()`:

```python
from parametrizani import TopologyGenerator

topo = TopologyGenerator('./work', force_field='gaff2')

# AM1-BCC charges (default, fast)
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')

# Gasteiger charges (fastest, least accurate)
amber_files = topo.generate_amber(conf['mol_file'], charge_method='gasteiger')
```

**Available `charge_method` values:**

| Value | Method | Description |
|-------|--------|-------------|
| `am1bcc` | AM1-BCC | Fast semi-empirical charges (default). Uses antechamber. |
| `gasteiger` | Gasteiger | Fastest, least accurate. Useful for quick tests. |

> **Recommendation:** Use `am1bcc` for most applications. For higher accuracy (charged molecules, hydrogen bonding), use Psi4 RESP charges as shown below.

### Psi4 RESP Charges (Optional)

For QM-quality RESP charges, use `calculate_resp_charges()` followed by `generate_amber()` with the `resp_charges_file` parameter:

```python
from parametrizani import calculate_resp_charges, TopologyGenerator, ConformerGenerator

# 1. Generate conformer
gen = ConformerGenerator(smiles, 'smiles', './work')
conf = gen.run()

# 2. Calculate RESP charges with Psi4 (requires: conda install -c conda-forge psi4 resp)
resp = calculate_resp_charges(
    conf['mol_file'],    # MOL, MOL2, SDF, or PDB file
    method='HF',         # QM method for ESP calculation
    basis='6-31G*',      # Basis set
    charge=0,            # Net molecular charge
    multiplicity=1,      # Spin multiplicity
)

print(f"RESP charges for {resp['num_atoms']} atoms:")
for symbol, q in zip(resp['atom_symbols'], resp['charges']):
    print(f"  {symbol:2s}  {q:8.4f}")
print(f"Total charge: {resp['charges'].sum():.4f}")

# 3. Generate GAFF2 topology using the Psi4 RESP charges
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(
    conf['mol_file'],
    resp_charges_file=resp['output_file']  # Uses Psi4 charges via antechamber -c rc
)
```

> **How it works:** `resp_charges_file` tells antechamber to read pre-computed charges
> from the file (`-c rc -cf <file>`) instead of computing them internally. This avoids
> the need for Gaussian/GAMESS that antechamber's built-in `-c resp` mode requires.

**Available methods for RESP:**

| Method | Description | Speed |
|--------|-------------|-------|
| `HF` | Hartree-Fock | Fast (standard for RESP) |
| `B3LYP` | B3LYP DFT functional | Medium |
| `MP2` | Møller-Plesset 2nd order | Slow (higher accuracy) |
| `wB97X-D` | Range-separated hybrid + dispersion | Medium |
| `PBE0` | PBE0 hybrid functional | Medium |
| `M06-2X` | M06-2X functional | Medium |

**Available basis sets:**

| Basis | Description | Recommended for |
|-------|-------------|-----------------|
| `6-31G*` | Pople double-zeta + polarization | Standard RESP (default, recommended) |
| `6-31G**` | Adds polarization on H atoms | Slightly better H charges |
| `cc-pVDZ` | Correlation-consistent double-zeta | General use |
| `cc-pVTZ` | Correlation-consistent triple-zeta | Higher accuracy (slower) |
| `aug-cc-pVDZ` | Augmented double-zeta | Anions, lone pairs |
| `def2-SVP` | Karlsruhe split-valence polarized | Fast alternative |
| `def2-TZVP` | Karlsruhe triple-zeta | High accuracy alternative |

> **Recommendation:** Use `HF/6-31G*` for standard RESP charges (matches the original Bayly et al. 1993 protocol). Use `B3LYP/cc-pVTZ` or `MP2/cc-pVDZ` for higher accuracy when needed.

### Key Parameters

| Parameter | Where | Description | Default |
|-----------|-------|-------------|---------|
| `fmax` | `scan_dihedral()` | Convergence threshold for geometry optimization (eV/Å) | 0.05 |
| `dihedral_indices` | Multiple | 0-based indices of the 4 atoms defining the dihedral | Required |
| `zero_dihedral` | `minimize_scan()` | Zero ALL torsion terms sharing the central bond | `True` |
| `n_terms` | `format_frcmod_params()` | Number of Fourier terms to include | `max_terms` |
| `idivf` | `format_frcmod_params()` | IDIVF scaling factor for equivalent torsions | 1 |
| `output_prefix` | `generate_amber()` | Prefix for output files (`'SYS'` or `'SYS_new'`) | `'SYS'` |


## Workflow

```
SMILES/PDB/MOL
     ↓
1. Conformer Generation (RDKit + MMFF94)
     ↓
2. Dihedral Scan (constrained rotation)
     ↓
3. Topology Generation & Atom Type Detection (antechamber + tLEaP)
     ↓
4. Reference Energy Calculation (TorchANI/MACE/xTB/AIMNet2, constrained optimization)
     ↓
5. MM Energy Minimization (OpenMM, dihedral zeroed around central bond)
     ↓
6. Dihedral Optimization (Least-squares fitting, 1–6 Fourier terms)
     ↓
7. Validation (RMSE, MAE, R², quality rating)
     ↓
8. Write FRCMOD & Update Topology (zero central-pair, delete exact, insert optimized)
     ↓
9. Generate Final Topology (AMBER/GROMACS/OpenMM with new parameters)
     ↓
10. Verification (MM scan with new topology, zero_dihedral=False)
```

## Quality Ratings

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | ≤ 0.25 | Near-QM accuracy |
| Good | ≤ 0.50 | Suitable for most applications |
| Acceptable | ≤ 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

## Bugs

- If you encounter any bugs, please report the issue to https://github.com/pablo-arantes/ParametrizANI/issues

## Acknowledgments

- ParametrizANI by **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Souvik Sinha** and **Giulia Palermo**
- We would like to thank the OpenMM team for developing an excellent and open source engine.
- We would like to thank the [Psi4](https://psicode.org/) team for developing an excellent and open source suite of ab initio quantum chemistry.
- We would like to thank the [Roitberg](https://roitberg.chem.ufl.edu/) team for developing the fantastic [TorchANI](https://github.com/aiqm/torchani).
- We would like to thank the [Xavier Barril](http://www.ub.edu/bl/) team for their protocol on dihedrals parametrization and for the genetic algorithm script.
- We would like to thank [iwatobipen](https://twitter.com/iwatobipen) for his fantastic [blog](https://iwatobipen.wordpress.com/) on chemoinformatics.
- Also, credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin.
- Finally, we would like to thank [Making it rain](https://github.com/pablo-arantes/making-it-rain) team, **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Marcelo D. Polêto** ([@mdpoleto](https://twitter.com/mdpoleto)), **Conrado Pedebos** ([@ConradoPedebos](https://twitter.com/ConradoPedebos)) and **Rodrigo Ligabue-Braun** ([@ligabue_braun](https://twitter.com/ligabue_braun)), for their amazing work.

## How should I reference this work?

- For **ParametrizANI**, please cite:
  Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules."
  *Journal of Chemical Information and Modeling* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)

- For **TorchANI**, please cite:
  Gao et al. "TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials."
  *Journal of Chemical Information and Modeling* (2020) doi: [10.1021/acs.jcim.0c00451](https://doi.org/10.1021/acs.jcim.0c00451)

- For **OpenMM**, please also cite:
  Eastman et al. "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics."
  *PLOS Computational Biology* (2017) doi: [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)

- For **Molecular Dynamics Notebook**, please also cite:
  Arantes et al. "Making it rain: cloud-based molecular simulations for everyone."
  *Journal of Chemical Information and Modeling* (2021) doi: [10.1021/acs.jcim.1c00998](https://doi.org/10.1021/acs.jcim.1c00998)

## License

MIT License - see [LICENSE](LICENSE) for details.
