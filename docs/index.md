# ParametrizANI

**Fast and Accessible Dihedral Parametrization for Small Molecules**

![ParametrizANI TOC](https://github.com/pablo-arantes/ParametrizANI/blob/main/TOC_graphic.png?raw=true)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

ParametrizANI is a production-ready Python package for dihedral parameter optimization using neural network potentials and molecular mechanics. It provides an end-to-end pipeline from SMILES strings to force field parameters compatible with **AMBER**, **GROMACS**, and **OpenMM**.

We have integrated **TorchANI2** into our pipeline, along with the latest improved ANI models trained on the expanded 2× dataset (**ANI-2xr, ANI-2dr, ANI-2xr-Snn, and ANI-mbis**). These models were trained at the B97-3c level of theory and include explicit repulsion and dispersion corrections, smoother potential energy surfaces (PES), and MBIS-derived charges.

Our goal is to democratize research by providing a research-friendly environment that is free from resource constraints, enabling teams of all sizes to perform dihedral parametrization with DFT-level accuracy.

## Key Features and Benefits

**• Robust Backbone:** ParametrizANI leverages TorchANI, a robust PyTorch-based deep learning program, as its benchmark to ensure precision in parametrization tasks. TorchANI is crucial for training and inference of ANI (ANAKIN-ME) deep learning models, which are fundamental for predicting potential energy surfaces and other molecular system attributes.

**• Accuracy and Efficiency:** ParametrizANI establishes detailed protocols for dihedral parametrization using both GAFF and OpenFF force fields. By integrating TorchANI's predictive power, ParametrizANI offers a streamlined and accurate approach to parametrization, especially for small molecules. TorchANI's neural network models predict molecular energies and properties with high accuracy and efficiency, significantly reducing computation time compared to traditional Quantum Mechanical (QM) methods.

**• Cloud-Based Accessibility:** The tool harnesses the power of Google Colaboratory (Colab), a hosted Jupyter Notebook service that provides free access to computing resources. This makes ParametrizANI a feasible, cost-effective, and accessible approach to compound parametrization, particularly beneficial for investigators worldwide, including those with limited resources. Our notebooks are designed to run efficiently on CPU cores, requiring no heavy parallel processing.

**• Comprehensive Workflow:** ParametrizANI provides comprehensive workflows implemented in Google Colab notebooks, exemplifying a complete pipeline for dihedral parametrization from SMILES strings generation to force field parameter optimization. These workflows enable researchers to efficiently perform accurate and reliable dihedral parametrization.

**• Versatile and Customizable:** The notebooks are designed for ease of use, following the Jupyter Notebook structure, with an initial configuration step taking less than 5 minutes. Users can select between GAFF and OpenFF force fields, choose charge calculation methods (AM1-BCC or RESP), and even upload their own reference energy profiles calculated using external software (e.g., Gaussian, GAMESS). This flexibility allows for customization to specific research requirements and professional use.

**• Broad Applicability:** ParametrizANI is not only suited for advanced molecular dynamics research and computational drug discovery but also serves as an excellent tool for educational purposes. It allows students to independently run the entire parametrization process without local software compilation or extensive coding experience, with embedded visualization at each step.

## Feature Summary

- **Multiple ML Potentials** — TorchANI (ANI-2x, ANI-1x, ANI-1ccx, ANI-2xr, ANI-2dr, ANI-2xr-Snn, ANI-mbis), AIMNet2 (wB97M-D3, B97-3c), MACE-OFF, GFN2-xTB
- **Force Field Support** — GAFF2 and OpenFF (Sage, SMIRNOFF) force fields
- **Complete Pipeline** — From SMILES to optimized topology in one script
- **Automatic Atom Typing** — Detects dihedral atom types from MOL2 files
- **Multi-Format Export** — AMBER (.prmtop/.crd), GROMACS (.top/.gro), OpenMM (.xml)
- **Psi4 RESP Charges** — Optional QM-quality RESP charges via Psi4 (HF, B3LYP, MP2)
- **Cloud-Ready** — Runs on Google Colab with free CPU resources
- **Validation** — Built-in quality metrics (RMSE, MAE, R², quality rating)

## Quick Example

```python
from parametrizani import (
    ConformerGenerator, TopologyGenerator,
    calculate_reference_energies, optimize_dihedral, get_dihedral_atom_types,
)

dihedral_indices = [0, 1, 2, 3]

gen = ConformerGenerator('CC(=O)OC', 'smiles', './work')
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)

ref = calculate_reference_energies(
    scan['conformers'], scan['angles'], method='torchani',
    dihedral_indices=dihedral_indices
)

opt = optimize_dihedral(ref['angles'], ref['energies_relative'], atom_types=atom_types)
print(f"RMSE: {opt['rmse']:.4f} kcal/mol")
```

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
|--------|---------|-----------------|
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

For detailed information on each model, see the [Supported Methods](methods.md) page.

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

## Psi4 RESP Charges (Optional)

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

**Available methods:**

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
| `6-31G*` | Pople double-zeta + polarization | Standard RESP (default) |
| `6-31G**` | Adds polarization on H atoms | Better H charges |
| `cc-pVDZ` | Correlation-consistent double-zeta | General use |
| `cc-pVTZ` | Correlation-consistent triple-zeta | Higher accuracy |
| `aug-cc-pVDZ` | Augmented double-zeta | Anions, lone pairs |
| `def2-SVP` | Karlsruhe split-valence polarized | Fast alternative |
| `def2-TZVP` | Karlsruhe triple-zeta | High accuracy alternative |

> **Recommendation:** Use `HF/6-31G*` for standard RESP charges (matches the original Bayly et al. 1993 protocol). Use `B3LYP/cc-pVTZ` or `MP2/cc-pVDZ` for higher accuracy when needed.

## Key Parameters

| Parameter | Where | Description | Default |
|-----------|-------|-------------|---------|
| `fmax` | `scan_dihedral()` | Convergence threshold for geometry optimization (eV/Å) | 0.05 |
| `dihedral_indices` | Multiple | 0-based indices of the 4 atoms defining the dihedral | Required |
| `zero_dihedral` | `minimize_scan()` | Zero ALL torsion terms sharing the central bond | `True` |
| `n_terms` | `format_frcmod_params()` | Number of Fourier terms to include | `max_terms` |
| `idivf` | `format_frcmod_params()` | IDIVF scaling factor for equivalent torsions | 1 |
| `output_prefix` | `generate_amber()` | Prefix for output files (`'SYS'` or `'SYS_new'`) | `'SYS'` |
| `resp_charges_file` | `generate_amber()` | Path to pre-computed RESP charges file (from Psi4) | `None` |

## Quality Ratings

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | ≤ 0.25 | Near-QM accuracy |
| Good | ≤ 0.50 | Suitable for most applications |
| Acceptable | ≤ 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

## Google Colab Notebooks

| Notebook | Description |
|----------|-------------|
| [**Notebook A** — GAFF2](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_GAFF2.ipynb) | Dihedral parametrization for GAFF force field |
| [**Notebook B** — OpenFF](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_OpenFF.ipynb) | Dihedral parametrization for OpenFF force fields |
| [**Notebook C** — TorchANI+Psi4](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_TorchANI%2BPsi4.ipynb) | Dihedral parametrization with Psi4 reference + TorchANI optimization |
| [**Notebook D** — RotProf](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RotProf.ipynb) | Rotational Profile fitting |
| [**Notebook E** — RESP Charges](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RESP_charges.ipynb) | Geometry optimization + Psi4 RESP charges → GAFF2 topology (+ optional dihedral parametrization) |

## Acknowledgments

- ParametrizANI by **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Souvik Sinha** and **Giulia Palermo**
- We would like to thank the OpenMM team for developing an excellent and open source engine.
- We would like to thank the [Psi4](https://psicode.org/) team for developing an excellent and open source suite of ab initio quantum chemistry.
- We would like to thank the [Roitberg](https://roitberg.chem.ufl.edu/) team for developing the fantastic [TorchANI](https://github.com/aiqm/torchani).
- We would like to thank the [Xavier Barril](http://www.ub.edu/bl/) team for their protocol on dihedrals parametrization and for the genetic algorithm script.
- We would like to thank [iwatobipen](https://twitter.com/iwatobipen) for his fantastic [blog](https://iwatobipen.wordpress.com/) on chemoinformatics.
- Also, credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin.
- Finally, we would like to thank [Making it rain](https://github.com/pablo-arantes/making-it-rain) team for their amazing work.

## Citation

If you use ParametrizANI in your research, please cite:

- **ParametrizANI:**
  Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules."
  *Journal of Chemical Information and Modeling* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)

- **TorchANI:**
  Gao et al. "TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials."
  *Journal of Chemical Information and Modeling* (2020) doi: [10.1021/acs.jcim.0c00451](https://doi.org/10.1021/acs.jcim.0c00451)

- **OpenMM:**
  Eastman et al. "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics."
  *PLOS Computational Biology* (2017) doi: [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)

## Bugs

If you encounter any bugs, please report the issue to [https://github.com/pablo-arantes/ParametrizANI/issues](https://github.com/pablo-arantes/ParametrizANI/issues)
