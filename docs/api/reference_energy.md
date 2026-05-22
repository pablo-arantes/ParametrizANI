# ReferenceEnergyCalculator

Calculate reference energies using ML potentials (TorchANI, AIMNet2, MACE-OFF, GFN2-xTB).

## Usage

```python
from parametrizani import ReferenceEnergyCalculator

calc = ReferenceEnergyCalculator('torchani', './work', device='cpu')
ref = calc.scan_dihedral(
    conformer_files, angles,
    optimize=True, fmax=0.001,
    dihedral_indices=[8, 9, 10, 15]
)
```

## API Reference

::: parametrizani.reference_energy.ReferenceEnergyCalculator
    options:
      members:
        - __init__
        - scan_dihedral
        - calculate_energy
      show_root_heading: true
      heading_level: 3

::: parametrizani.reference_energy.AIMNet2Calculator
    options:
      members:
        - __init__
        - calculate
        - set_charge
      show_root_heading: true
      heading_level: 3
