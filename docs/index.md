# ParametrizANI

**Fast and Accessible Dihedral Parametrization for Small Molecules**

![ParametrizANI TOC](https://github.com/pablo-arantes/ParametrizANI/blob/main/TOC_graphic.png?raw=true)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

ParametrizANI is a production-ready Python package for dihedral parameter optimization using neural network potentials and molecular mechanics. It provides an end-to-end pipeline from SMILES strings to force field parameters compatible with **AMBER**, **GROMACS**, and **OpenMM**.

Our goal is to democratize research by providing a research-friendly environment that is free from resource constraints, enabling teams of all sizes to perform dihedral parametrization with DFT-level accuracy.

## Key Features

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

## Psi4 RESP Charges (Optional)

```python
from parametrizani import calculate_resp_charges, TopologyGenerator

# Requires: conda install -c conda-forge psi4 resp
resp = calculate_resp_charges('molecule.mol', method='HF', basis='6-31G*')
print(resp['charges'])  # RESP charges array

# Use Psi4 RESP charges in topology generation
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber('molecule.mol', resp_charges_file=resp['output_file'])
```

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

## Google Colab Notebooks

| Notebook | Description |
|----------|-------------|
| [**Notebook A** — GAFF2](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_GAFF2.ipynb) | Dihedral parametrization for GAFF force field |
| [**Notebook B** — OpenFF](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_OpenFF.ipynb) | Dihedral parametrization for OpenFF force fields |
| [**Notebook C** — TorchANI+Psi4](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_TorchANI%2BPsi4.ipynb) | Dihedral parametrization with Psi4 reference + TorchANI optimization |
| [**Notebook D** — RotProf](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RotProf.ipynb) | Rotational Profile fitting |
| [**Notebook E** — RESP Charges](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RESP_charges.ipynb) | Geometry optimization + Psi4 RESP charges → GAFF2 topology (+ optional dihedral parametrization) |

## Citation

If you use ParametrizANI in your research, please cite:

> Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules."
> *Journal of Chemical Information and Modeling* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)
