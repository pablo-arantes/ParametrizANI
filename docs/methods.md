# Supported Methods

## ML Potentials for Reference Energies

The `method` parameter in `ReferenceEnergyCalculator` selects the ML potential:

```python
from parametrizani import ReferenceEnergyCalculator

# Use any of the supported method strings:
calc = ReferenceEnergyCalculator('ani2xr', './work')  # TorchANI ANI-2xr
```

### TorchANI Models

All ANI models are loaded from the [torchani](https://github.com/aiqm/torchani) package.

| `method` string | Model | Level of Theory | Elements |
|-----------------|-------|-----------------|----------|
| `torchani` or `ani2x` | ANI-2x | ωB97X/6-31G* | H, C, N, O, F, S, Cl |
| `ani1x` | ANI-1x | ωB97X/6-31G* | H, C, N, O |
| `ani1ccx` | ANI-1ccx | CCSD(T)/CBS | H, C, N, O |
| `ani2xr` | ANI-2xr | B97-3c | H, C, N, O, F, S, Cl |
| `ani2dr` | ANI-2dr | B97-3c | H, C, N, O, F, S, Cl |
| `ani2xr_snn` | ANI-2xr-Snn | B97-3c | H, C, N, O, F, S, Cl |
| `ani_mbis` | ANI-mbis | B97-3c | H, C, N, O, F, S, Cl |

!!! tip "Recommended model"
    For most organic molecules, **ANI-2xr** provides excellent accuracy with broad element
    coverage. Use **ANI-1ccx** for highest accuracy on HCNO-only molecules.

### AIMNet2 Models

AIMNet2 models use `torch.jit` (`.jpt` files) from [pablo-arantes/AIMNet2](https://github.com/pablo-arantes/AIMNet2).

| `method` string | Model | Level of Theory | Elements |
|-----------------|-------|-----------------|----------|
| `wb97m_d3` or `aimnet2` | AIMNet2 (wB97M-D3) | ωB97M-D3 | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |
| `b97_3c` | AIMNet2 (B97-3c) | B97-3c | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |

!!! note "Installation"
    AIMNet2 requires cloning the model repository:
    ```bash
    git clone https://github.com/pablo-arantes/AIMNet2.git
    ```
    The package automatically searches for models in `./AIMNet2/models/`.

### MACE-OFF

| `method` string | Model | Training Data |
|-----------------|-------|---------------|
| `mace` | MACE-OFF (medium) | DFT (SPICE dataset) |

Supported elements: H, C, N, O, F, P, S, Cl, Br, I.

### GFN2-xTB

| `method` string | Model | Method |
|-----------------|-------|--------|
| `xtb` or `gfn2xtb` | GFN2-xTB | Semi-empirical tight-binding |

Requires: `conda install -c conda-forge xtb-python`

### How to select the model in code

```python
from parametrizani import ReferenceEnergyCalculator

# Use any of the method strings from the tables above
calc = ReferenceEnergyCalculator('ani2xr', './work')  # TorchANI ANI-2xr
calc = ReferenceEnergyCalculator('aimnet2', './work')  # AIMNet2 wB97M-D3
calc = ReferenceEnergyCalculator('mace', './work')     # MACE-OFF
calc = ReferenceEnergyCalculator('xtb', './work')      # GFN2-xTB
```

In the Colab notebooks, select the model from the dropdown widget in the "Reference Energy Calculation" cell.

---

## Force Fields

### GAFF2 (General AMBER Force Field 2)

Used via `TopologyGenerator` + `EnergyMinimizer`:

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

- Requires AmberTools (antechamber, tleap, parmchk2)
- Generates .prmtop/.crd topology files
- Full export to GROMACS and OpenMM formats

### OpenFF (Open Force Field)

Used directly via `EnergyMinimizer.minimize_scan_openff()`:

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

**Available OpenFF versions:**

| Value | Force Field | Notes |
|-------|-------------|-------|
| `openff-2.0.0` | OpenFF 2.0.0 (Sage) | Recommended |
| `openff-1.3.1` | OpenFF 1.3.1 | Legacy |
| `openff-1.2.0` | OpenFF 1.2.0 | Legacy |
| `smirnoff99Frosst-1.1.0` | SMIRNOFF99Frosst | Original SMIRNOFF |

!!! info "Key difference"
    GAFF2 requires generating topology files first (`generate_amber()`), while OpenFF
    works directly from the SMILES string — no separate topology step needed for minimization.

In the Colab notebooks, use **Notebook A** for GAFF2 or **Notebook B** for OpenFF.

---

## Charge Methods

### AM1-BCC and Gasteiger

The charge method is set in `TopologyGenerator.generate_amber()`:

```python
from parametrizani import TopologyGenerator

topo = TopologyGenerator('./work', force_field='gaff2')

# AM1-BCC charges (default, fast)
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')

# Gasteiger charges (fastest, least accurate)
amber_files = topo.generate_amber(conf['mol_file'], charge_method='gasteiger')
```

| Value | Method | Speed | Accuracy | Notes |
|-------|--------|-------|----------|-------|
| `am1bcc` | AM1-BCC | Fast | Good | Default. Uses antechamber. |
| `gasteiger` | Gasteiger | Very fast | Low | Useful for quick tests only. |

!!! tip "Recommendation"
    Use `am1bcc` for most applications. For higher accuracy (charged molecules,
    hydrogen bonding), use Psi4 RESP charges as shown below.

### Psi4 RESP Charges

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

See **Notebook E** (RESP Charges) for the complete workflow.

---

## Key Parameters Reference

| Parameter | Location | Description | Default |
|-----------|----------|-------------|---------|
| `fmax` | `scan_dihedral()` | Convergence threshold for geometry optimization (eV/Å) | 0.05 |
| `dihedral_indices` | Multiple | 0-based indices of the 4 atoms defining the dihedral | Required |
| `zero_dihedral` | `minimize_scan()` | Zero ALL torsion terms sharing the central bond | `True` |
| `max_terms` | `DihedralOptimizer()` | Maximum number of Fourier terms (1–6) | 6 |
| `n_terms` | `format_frcmod_params()` | Number of terms to write to FRCMOD | `max_terms` |
| `idivf` | `format_frcmod_params()` | IDIVF scaling factor for equivalent torsions | 1 |
| `output_prefix` | `generate_amber()` | Prefix for output files | `'SYS'` |
| `force_constant` | `minimize_scan()` | Dihedral restraint force constant (kcal/mol/rad²) | 1000 |
| `opt_tol` | `minimize_scan()` | MM minimization tolerance (kJ/mol/nm) | 0.001 |
| `charge_method` | `generate_amber()` | Charge calculation method | `'am1bcc'` |
| `resp_charges_file` | `generate_amber()` | Path to pre-computed RESP charges file | `None` |

---

## Quality Ratings

The `ParameterValidator` assigns quality ratings based on RMSE between reference and fitted energies:

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | ≤ 0.25 | Near-QM accuracy |
| Good | ≤ 0.50 | Suitable for most applications |
| Acceptable | ≤ 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

!!! tip "Tips for improving quality"
    - Increase `max_terms` (up to 6 Fourier terms)
    - Use a smaller scan step size (e.g., 10° or 15° instead of 30°)
    - Try a different ML potential (e.g., ANI-2xr instead of ANI-2x)
    - Reduce `fmax` for tighter geometry optimization convergence
