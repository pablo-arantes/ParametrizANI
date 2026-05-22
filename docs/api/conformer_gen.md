# ConformerGenerator

Generate 3D conformers from SMILES, PDB, or MOL files with RDKit.

## Usage

```python
from parametrizani import ConformerGenerator

gen = ConformerGenerator('CC(=O)OC', 'smiles', './work')
conf = gen.run()
scan = gen.generate_dihedral_conformers([0, 1, 2, 3], step=30)
```

## API Reference

::: parametrizani.conformer_gen.ConformerGenerator
    options:
      members:
        - __init__
        - run
        - generate_dihedral_conformers
      show_root_heading: true
      heading_level: 3
