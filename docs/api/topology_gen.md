# TopologyGenerator

Generate topology files for MD simulations in AMBER, GROMACS, and OpenMM formats.

## Usage

```python
from parametrizani import TopologyGenerator

topo = TopologyGenerator('./work', force_field='gaff2')

# Generate initial topology
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')

# Update FRCMOD with optimized parameters
updated_frcmod = topo.update_frcmod(
    amber_files['frcmod'],
    optimized_params,
    dihedral_indices=dihedral_indices,
    mol2_file=mol2_file,
)

# Regenerate topology with new parameters
new_amber = topo.generate_amber(
    conf['mol_file'],
    frcmod_file=updated_frcmod,
    mol2_file=mol2_file,
    output_prefix='SYS_new',
)

# Export to other formats
gromacs_files = topo.generate_gromacs(new_amber['prmtop'], new_amber['inpcrd'])
openmm_files = topo.generate_openmm(new_amber['prmtop'], new_amber['inpcrd'])
```

## FRCMOD Update Logic

The `update_frcmod()` method follows the original ParametrizANI notebook algorithm:

1. **Zero** all DIHE lines where the central 2-atom segment matches (sets PK=0, PHASE=0, PN=1)
2. **Delete** DIHE lines matching the full 4-atom pattern (forward or reversed)
3. **Insert** new optimized parameter lines immediately after the DIHE header

When `dihedral_indices` and `mol2_file` are provided, patterns are derived from the
actual MOL2 atom types (matching the original notebook's `process_mol2_elements()` logic).

## API Reference

::: parametrizani.topology_gen.TopologyGenerator
    options:
      members:
        - __init__
        - generate_amber
        - generate_gromacs
        - generate_openmm
        - generate_all
        - update_frcmod
        - package_results
      show_root_heading: true
      heading_level: 3
