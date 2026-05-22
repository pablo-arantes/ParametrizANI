# psi4_utils

::: parametrizani.psi4_utils
    options:
      show_source: false
      heading_level: 2
      members:
        - calculate_resp_charges

## Usage Example

```python
from parametrizani import calculate_resp_charges

# Calculate RESP charges using Psi4
result = calculate_resp_charges(
    'molecule.mol',
    method='HF',          # QM method for ESP calculation
    basis='6-31G*',       # Basis set
    charge=0,
    multiplicity=1,
    memory='2 GB',
    n_threads=2,
)

print(result['charges'])       # np.ndarray of RESP charges
print(result['atom_symbols'])  # ['C', 'C', 'O', 'H', ...]
print(result['output_file'])   # Path to saved charges file
```

## Requirements

This module requires Psi4 and the resp package:

```bash
conda install -c conda-forge psi4 resp -y
```

These are **optional** dependencies — they are NOT required for the main ParametrizANI workflow.

## Supported Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `HF` | Hartree-Fock | Fast (standard for RESP) |
| `B3LYP` | B3LYP DFT functional | Medium |
| `MP2` | Møller-Plesset 2nd order perturbation theory | Slow (higher accuracy) |
| `wB97X-D` | Range-separated hybrid with dispersion | Medium |
| `PBE0` | PBE0 hybrid functional | Medium |
| `M06-2X` | M06-2X functional | Medium |

> **Recommendation:** Use `HF` for standard RESP charges (matches the original Bayly et al. 1993 protocol and the AMBER force field convention). Use `B3LYP` or `MP2` when higher accuracy is needed.

## Supported Basis Sets

| Basis | Description | Recommended for |
|-------|-------------|-----------------|
| `6-31G*` | Pople double-zeta + polarization | Standard RESP (default, recommended) |
| `6-31G**` | Adds polarization on H atoms | Slightly better H charges |
| `cc-pVDZ` | Correlation-consistent double-zeta | General use |
| `cc-pVTZ` | Correlation-consistent triple-zeta | Higher accuracy (slower) |
| `aug-cc-pVDZ` | Augmented double-zeta | Anions, lone pairs |
| `def2-SVP` | Karlsruhe split-valence polarized | Fast alternative |
| `def2-TZVP` | Karlsruhe triple-zeta valence polarized | High accuracy alternative |

> **Recommendation:** Use `6-31G*` for standard RESP charges. This is the basis set used in the original RESP paper (Bayly et al. 1993) and provides the best compatibility with AMBER force fields.

## RESP Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resp_a` | Restraint strength (hyperbolic penalty) | 0.0005 |
| `resp_b` | Tightness of the hyperbolic penalty | 0.1 |
| `charge` | Net molecular charge | 0 |
| `multiplicity` | Spin multiplicity (1=singlet, 2=doublet, ...) | 1 |
| `memory` | Memory for Psi4 calculation | `'2 GB'` |
| `n_threads` | Number of CPU threads | 2 |

## Integration with GAFF2 Workflow

After computing RESP charges, pass the charges file to `generate_amber()`:

```python
from parametrizani import calculate_resp_charges, TopologyGenerator, ConformerGenerator

# 1. Generate conformer
gen = ConformerGenerator(smiles, 'smiles', './work')
conf = gen.run()

# 2. Calculate RESP charges with Psi4
resp_result = calculate_resp_charges(conf['mol_file'], method='HF', basis='6-31G*')
print(f"Charges: {resp_result['charges']}")
print(f"Saved to: {resp_result['output_file']}")

# 3. Generate topology using the Psi4 RESP charges
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(
    conf['mol_file'],
    resp_charges_file=resp_result['output_file']  # Reads pre-computed Psi4 charges
)
```

The `resp_charges_file` parameter tells antechamber to read pre-computed charges from
the file (`-c rc -cf <file>`) rather than trying to compute RESP charges internally
(which would require Gaussian/GAMESS). This is the correct way to integrate Psi4 RESP
charges into the AMBER topology generation pipeline.
