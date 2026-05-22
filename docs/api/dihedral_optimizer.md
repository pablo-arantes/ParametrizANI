# DihedralOptimizer

Optimize dihedral parameters via least-squares fitting of Fourier series coefficients.

## Usage

```python
from parametrizani import DihedralOptimizer

optimizer = DihedralOptimizer(max_terms=6, work_dir='./work')
result = optimizer.run_optimization(
    ref_angles, ref_energies,
    mm_energies=mm_energies,
    atom_types=['ca', 'ca', 'os', 'ca']
)

# Select specific number of terms
params = optimizer.format_frcmod_params(result, n_terms=3, idivf=1)
print(params)

# Write FRCMOD file
frcmod_file = optimizer.write_frcmod(result, idivf=1, n_terms=3)
```

## FRCMOD Format

The optimizer writes AMBER-format FRCMOD lines with continuation signs:

```
DIHE
ca-ca-os-ca   1       1.2000    180.00    -2.0
ca-ca-os-ca   1       0.3000      0.00     3.0
```

- Negative PN (periodicity) indicates continuation (more terms follow)
- Positive PN marks the final term for that dihedral
- IDIVF is the scaling factor for equivalent torsions

## API Reference

::: parametrizani.dihedral_optimizer.DihedralOptimizer
    options:
      members:
        - __init__
        - run_optimization
        - format_frcmod_params
        - get_parameters_for_terms
        - write_frcmod
        - calculate_dihedral_energy
      show_root_heading: true
      heading_level: 3
