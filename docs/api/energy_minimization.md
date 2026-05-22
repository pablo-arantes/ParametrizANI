# EnergyMinimizer

OpenMM-based energy minimization with dihedral restraints. Supports GAFF2 and OpenFF force fields.

## Usage

```python
from parametrizani import EnergyMinimizer

# GAFF2 workflow
minimizer = EnergyMinimizer('gaff2', './work')
mm = minimizer.minimize_scan(
    topology_file, coordinate_file,
    conformer_files, dihedral_indices,
    angles=angles, zero_dihedral=True,
    force_constant=1000, opt_tol=0.001,
)

# OpenFF workflow
minimizer = EnergyMinimizer('openff-2.0.0', './work')
mm = minimizer.minimize_scan_openff(
    smiles, conformer_files, dihedral_indices,
    angles=angles, zero_dihedral=True,
)
```

!!! important "Central bond zeroing"
    When `zero_dihedral=True`, ALL torsion terms sharing the central bond (atom2–atom3)
    are zeroed — not just the exact 4-atom quartet. This ensures consistency with the
    FRCMOD modification that operates on atom-type patterns.

## API Reference

::: parametrizani.energy_minimization.EnergyMinimizer
    options:
      members:
        - __init__
        - minimize_scan
        - minimize_scan_openff
      show_root_heading: true
      heading_level: 3
