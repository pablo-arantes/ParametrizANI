# Supported Methods

## ML Potentials for Reference Energies

The `method` parameter in `ReferenceEnergyCalculator` selects the ML potential:

```python
calc = ReferenceEnergyCalculator('ani2xr', './work', device='cpu')
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

---

## Force Fields

### GAFF2 (General AMBER Force Field 2)

Used via `TopologyGenerator` + `EnergyMinimizer`:

```python
topo = TopologyGenerator('./work', force_field='gaff2')
minimizer = EnergyMinimizer('gaff2', './work')
```

- Requires AmberTools (antechamber, tleap, parmchk2)
- Generates .prmtop/.crd topology files
- Full export to GROMACS and OpenMM formats

### OpenFF (Open Force Field)

Used directly via `EnergyMinimizer.minimize_scan_openff()`:

```python
minimizer = EnergyMinimizer('openff-2.0.0', './work')
mm = minimizer.minimize_scan_openff(smiles, conformers, dihedral_indices, ...)
```

**Available OpenFF versions:**

| Value | Force Field |
|-------|-------------|
| `openff-2.0.0` | OpenFF 2.0.0 (Sage) — Recommended |
| `openff-1.3.1` | OpenFF 1.3.1 |
| `openff-1.2.0` | OpenFF 1.2.0 |
| `smirnoff99Frosst-1.1.0` | SMIRNOFF99Frosst |

!!! info "Key difference"
    GAFF2 requires generating topology files first (`generate_amber()`), while OpenFF
    works directly from the SMILES string — no separate topology step needed for minimization.

---

## Charge Methods

The charge method is set in `TopologyGenerator.generate_amber()`:

```python
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')
```

| Value | Method | Speed | Accuracy | Notes |
|-------|--------|-------|----------|-------|
| `am1bcc` | AM1-BCC | Fast | Good | Default. Uses antechamber. |
| `resp` | RESP | Slow | High | Requires Psi4 or pre-computed charges. |
| `gasteiger` | Gasteiger | Very fast | Low | Useful for quick tests only. |

!!! tip "Recommendation"
    Use `am1bcc` for most applications. Use `resp` for charged molecules or when
    hydrogen bonding accuracy is critical. See **Notebook C** (TorchANI+Psi4) for RESP.

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
