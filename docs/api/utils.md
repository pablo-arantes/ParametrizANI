# Utilities

Helper functions for energy file I/O, atom type detection, and data processing.

## Usage

```python
from parametrizani import (
    read_energy_file,
    write_energy_file,
    get_dihedral_atom_types,
    extract_atom_info_from_pdb,
    relative_energies,
)

# Read/write energy profiles
angles, energies = read_energy_file('profile.dat')
write_energy_file('output.dat', angles, energies)

# Detect atom types from MOL2
atom_types = get_dihedral_atom_types('ligand.mol2', [8, 9, 10, 15])
# Returns e.g. ['ca', 'ca', 'os', 'ca']

# Extract atom info from PDB
atom_info = extract_atom_info_from_pdb('molecule.pdb')
```

## API Reference

::: parametrizani.utils
    options:
      members:
        - read_energy_file
        - write_energy_file
        - relative_energies
        - get_dihedral_atom_types
        - get_atom_types_from_mol2
        - extract_atom_info_from_pdb
        - create_work_dir
      show_root_heading: true
      heading_level: 3
