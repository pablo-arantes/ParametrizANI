# Installation

## Requirements

- Python 3.9+
- conda (for AmberTools, OpenMM, RDKit, xtb-python)

## Step 1: Install conda dependencies

Several packages are only available via conda-forge:

```bash
conda install -c conda-forge ambertools openmm openmmforcefields rdkit openbabel openff-toolkit xtb-python -y
```

## Step 2: Install ParametrizANI

```bash
git clone https://github.com/pablo-arantes/ParametrizANI.git
cd ParametrizANI
pip install -e .
```

## Step 3: Install ML potentials

```bash
# TorchANI (required for ANI models)
pip install torch torchani

# MACE-OFF (optional)
pip install mace-torch e3nn==0.4.4

# AIMNet2 (optional) - uses torch.jit models from the cloned repo
git clone https://github.com/pablo-arantes/AIMNet2.git
```

!!! note "AIMNet2 uses the older torch.jit interface"
    ParametrizANI uses the original AIMNet2 implementation with `torch.jit.load()` models
    (`.jpt` files) from the cloned repository. Do **not** use `pip install aimnet2` — that
    is a newer, incompatible version.

## Step 4 (Optional): Install Psi4 for RESP charges

Psi4 enables high-accuracy QM-based RESP charge calculation as an alternative to AM1-BCC:

```bash
conda install -c conda-forge psi4 resp -y
```

!!! note "Psi4 is optional"
    Psi4 is **not** required for the main ParametrizANI workflow. The default AM1-BCC
    charges from antechamber work well for most applications. Use Psi4 RESP when you need
    publication-quality charges computed at a specific level of theory (HF/6-31G*, B3LYP, MP2).

**On Google Colab**, Psi4 requires a separate conda environment:

```bash
mamba create -n psi4_env python=3.11 psi4 resp -c conda-forge --yes
source activate psi4_env
```

## Quick install (all pip extras)

```bash
pip install -e ".[full]"
```

!!! warning
    The `[full]` extra installs only pip-available packages (`torchani`, `mace-torch`,
    `parmed`, `ase`). You still need conda for `openmm`, `openff-toolkit`, `ambertools`,
    `rdkit`, and `xtb-python`.

## Google Colab

On Colab, everything is handled automatically via condacolab. Simply open one of the
provided notebooks and run the first two installation cells:

1. **Cell 1** installs condacolab + miniforge
2. **Cell 2** installs all dependencies via mamba/pip
3. **Cell 3** imports ParametrizANI

No local installation required.

## Verifying the installation

```python
import parametrizani
print(parametrizani.__version__)  # Should print "1.0.0"

from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
)
print("✓ All modules imported successfully!")

# Optional: verify Psi4
try:
    from parametrizani import calculate_resp_charges
    import psi4
    print("✓ Psi4 RESP available!")
except ImportError:
    print("  Psi4 not installed (optional)")
```

## Troubleshooting

### torchvision circular import error

If you see `AttributeError: partially initialized module 'torchvision' has no attribute 'extension'`:

```python
# Add this BEFORE importing parametrizani:
try:
    import torchvision
except (ImportError, AttributeError):
    pass
```

This is a known issue when `mace-torch` is installed alongside OpenFF on Colab.

### antechamber / tleap not found

Make sure AmberTools is installed via conda:

```bash
conda install -c conda-forge ambertools -y
```

### AIMNet2 model not found

Clone the models repository:

```bash
git clone https://github.com/pablo-arantes/AIMNet2.git
```

The package searches for models in `./AIMNet2/models/`, `./work/AIMNet2/models/`, and `/content/AIMNet2/models/` (Colab).

### Psi4 import error

If `calculate_resp_charges()` raises `ImportError`, install Psi4:

```bash
conda install -c conda-forge psi4 resp -y
```

Note: Psi4 is a large package (~1 GB). It is NOT required for the core ParametrizANI workflow.
