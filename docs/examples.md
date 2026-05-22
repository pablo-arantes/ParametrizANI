# Examples

## Example 1: Basic Fitting from Pre-computed Energies

Use this when you already have reference and MM energy profiles from external software
(Gaussian, GAMESS, etc.):

```python
from parametrizani import optimize_dihedral, validate_parameters, read_energy_file
from parametrizani import DihedralOptimizer
import numpy as np
import matplotlib.pyplot as plt

# Read energy files (format: angle energy, one per line)
ref_angles, ref_energies = read_energy_file('qm.dat')
mm_angles, mm_energies = read_energy_file('mm.dat')

# Fit dihedral parameters
max_terms = 6
IDIVF = 1
atom_types = ['ca', 'ca', 'os', 'ca']

result = optimize_dihedral(
    ref_angles, ref_energies,
    atom_types=atom_types,
    mm_energies=mm_energies,
    max_terms=max_terms,
    work_dir='./output'
)

# --- Print RMSE per number of terms ---
print(f"RMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")

# --- Print optimized FRCMOD parameters ---
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# Choose fewer terms if desired
optimizer = DihedralOptimizer(max_terms=max_terms, work_dir='./output')
params_3_terms = optimizer.format_frcmod_params(result, n_terms=3, idivf=IDIVF)
print(f"\n3-term FRCMOD Parameters:")
print(params_3_terms)

# Validate
val = validate_parameters(ref_angles, ref_energies, result['best_fit'], './output')
print(f"\nQuality: {val['quality']} (R²: {val['r_squared']:.4f})")

# --- Plot: Optimized Profile ---
plt.figure(figsize=(8, 5))
plt.plot(ref_angles, ref_energies, 'o-', linewidth=1.5, label='Reference (QM)')
plt.plot(mm_angles, mm_energies, 's-', linewidth=1.5, label='MM (dihedral zeroed)')
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label='Optimized')
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Example 2: GAFF2 with ANI-2xr

Full parametrization using the newer ANI-2xr model trained at B97-3c level:

```python
from parametrizani import (
    ConformerGenerator, ReferenceEnergyCalculator, EnergyMinimizer,
    DihedralOptimizer, ParameterValidator, TopologyGenerator, get_dihedral_atom_types,
)
import os
import numpy as np
import matplotlib.pyplot as plt

smiles = 'c1ccc(cc1)Oc2ccccc2'  # Diphenyl ether
dihedral_indices = [3, 4, 11, 12]
workDir = './diphenyl_ether'
model_name = 'ani2xr'

gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)

# Use ANI-2xr (B97-3c level)
calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=0.001,
    dihedral_indices=dihedral_indices
)

minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True,
)

# --- Plot: Reference vs MM ---
min_deg, max_deg = min(ref['angles']), max(ref['angles'])
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label="GAFF2")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference vs GAFF2 Energy Profile')
plt.savefig(f'{model_name}_vs_gaff2.png', dpi=300, bbox_inches='tight')
plt.show()

optimizer = DihedralOptimizer(max_terms=4, work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)

# --- Print RMSE and parameters ---
print(f"\nRMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# --- Plot: Optimized Profile ---
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.0, label="GAFF2 (original)", alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label="Optimized")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# Write and apply
selected_params = optimizer.format_frcmod_params(result, n_terms=4, idivf=1)
optimizer.write_frcmod(result, idivf=1, n_terms=4)

topology_mol2 = os.path.join(topo.tleap_dir, 'new.mol2')
if not os.path.exists(topology_mol2):
    topology_mol2 = amber_files['mol2']

updated_frcmod = topo.update_frcmod(
    amber_files['frcmod'], selected_params,
    dihedral_indices=dihedral_indices, mol2_file=topology_mol2,
)

new_amber = topo.generate_amber(
    conf['mol_file'], frcmod_file=updated_frcmod,
    mol2_file=topology_mol2, output_prefix='SYS_new',
)

# Export GROMACS
gromacs = topo.generate_gromacs(new_amber['prmtop'], new_amber['inpcrd'])
print(f"GROMACS topology: {gromacs['top']}")

# Verify: re-run MM scan with new topology
mm_new = minimizer.minimize_scan(
    new_amber['prmtop'], new_amber['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=False,
)

# --- Plot: Verification ---
plt.figure(figsize=(10, 6))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=2, markersize=6,
         label=model_name + ' (Reference)')
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.5,
         label='GAFF2 (original, dihedral=0)', alpha=0.6)
plt.plot(result['angles'], result['best_fit'], 'D--', linewidth=1.5,
         label='Fitted curve', alpha=0.7)
plt.plot(mm_new['angles'], mm_new['energies_relative'], '^-', linewidth=2,
         markersize=6, label='GAFF2 (new parameters)', color='green')
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)', fontsize=12)
plt.ylabel('Relative Energy (kcal/mol)', fontsize=12)
plt.legend(frameon=True, fontsize=10)
plt.title('Verification: Optimized Parameters in Final Topology', fontsize=13)
plt.tight_layout()
plt.savefig('verification_final.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print verification RMSE ---
rmse_new = np.sqrt(np.mean(
    (np.array(ref['energies_relative']) - np.array(mm_new['energies_relative']))**2
))
print(f"\nRMSE (Reference vs New Topology): {rmse_new:.4f} kcal/mol")
if rmse_new < 1.0:
    print("  Parameters validated successfully!")
else:
    print("  RMSE > 1.0 - consider adjusting the number of Fourier terms")
```

---

## Example 3: OpenFF Workflow

For OpenFF, no topology generation is needed for MM minimization:

```python
from parametrizani import (
    ConformerGenerator, ReferenceEnergyCalculator, EnergyMinimizer,
    DihedralOptimizer, ParameterValidator,
)
import numpy as np
import matplotlib.pyplot as plt

smiles = 'CC(=O)Oc1ccccc1'  # Phenyl acetate
dihedral_indices = [1, 2, 3, 4]
workDir = './phenyl_acetate_openff'
model_name = 'torchani'

gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=0.001,
    dihedral_indices=dihedral_indices
)

# OpenFF minimization — no topology step needed
minimizer = EnergyMinimizer('openff-2.0.0', workDir)
mm = minimizer.minimize_scan_openff(
    smiles, scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True,
)

# --- Plot: Reference vs OpenFF ---
min_deg, max_deg = min(ref['angles']), max(ref['angles'])
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label="OpenFF 2.0.0")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference vs OpenFF Energy Profile')
plt.savefig(f'{model_name}_vs_openff.png', dpi=300, bbox_inches='tight')
plt.show()

optimizer = DihedralOptimizer(max_terms=6, work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
)

# --- Print RMSE and parameters ---
print(f"\nRMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# --- Plot: Optimized Profile ---
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.0, label="OpenFF (original)", alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label="Optimized")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'OpenFF Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile_openff.png', dpi=300, bbox_inches='tight')
plt.show()

validator = ParameterValidator(workDir)
val = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"\nQuality: {val['quality']} (RMSE: {val['rmse']:.4f} kcal/mol)")
```

---

## Example 4: AIMNet2 with MACE Comparison

Compare different ML potentials on the same molecule:

```python
from parametrizani import ConformerGenerator, ReferenceEnergyCalculator
import numpy as np
import matplotlib.pyplot as plt

smiles = 'c1ccc(cc1)C(=O)O'  # Benzoic acid
dihedral_indices = [3, 6, 7, 8]
workDir = './benzoic_acid'

gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=15)

methods = ['torchani', 'ani2xr', 'wb97m_d3', 'mace']
results = {}

for method in methods:
    calc = ReferenceEnergyCalculator(method, workDir)
    ref = calc.scan_dihedral(
        scan['conformers'], scan['angles'],
        optimize=True, fmax=0.001,
        dihedral_indices=dihedral_indices
    )
    results[method] = ref

# Plot comparison
plt.figure(figsize=(10, 6))
for method, ref in results.items():
    plt.plot(ref['angles'], ref['energies_relative'], 'o-', label=method)
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend()
plt.title('ML Potential Comparison')
plt.savefig('ml_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```
---

## Example 5: Psi4 RESP Charges with GAFF2 Workflow

Use QM-quality RESP charges (from Psi4) instead of the default AM1-BCC charges.
This provides more accurate partial charges, especially for polar molecules or
those with strong hydrogen bonding:

```python
from parametrizani import (
    ConformerGenerator, ReferenceEnergyCalculator, EnergyMinimizer,
    DihedralOptimizer, ParameterValidator, TopologyGenerator,
    get_dihedral_atom_types, calculate_resp_charges,
)
import os
import numpy as np
import matplotlib.pyplot as plt

smiles = 'CC(=O)Nc1ccccc1'  # Acetanilide
dihedral_indices = [1, 2, 3, 4]
workDir = './acetanilide_resp'
model_name = 'torchani'

# 1. Generate conformer
gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

# 2. Calculate RESP charges with Psi4 (requires: conda install -c conda-forge psi4 resp)
resp_result = calculate_resp_charges(
    conf['mol_file'],
    method='HF',           # QM method: 'HF', 'B3LYP', 'MP2', 'wB97X-D'
    basis='6-31G*',        # Basis set
    charge=0,              # Net molecular charge
    multiplicity=1,        # Spin multiplicity
    memory='4 GB',         # Psi4 memory
    n_threads=4,           # CPU threads
)

print(f"RESP charges calculated ({resp_result['method']}/{resp_result['basis']}):")
for symbol, q in zip(resp_result['atom_symbols'], resp_result['charges']):
    print(f"  {symbol:2s}  {q:8.4f}")
print(f"Total charge: {resp_result['charges'].sum():.4f}")
print(f"Saved to: {resp_result['output_file']}")

# 3. Generate topology with Psi4 RESP charges
topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(
    conf['mol_file'],
    resp_charges_file=resp_result['output_file']  # Uses Psi4 charges via antechamber -c rc
)
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
print(f"\nAtom types: {atom_types}")

# 4. Reference energy (ML potential)
calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=0.001,
    dihedral_indices=dihedral_indices
)

# 5. MM minimization
minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True,
)

# --- Plot: Reference vs MM ---
min_deg, max_deg = min(ref['angles']), max(ref['angles'])
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label="GAFF2 (RESP)")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference vs GAFF2 (RESP charges)')
plt.savefig(f'{model_name}_vs_gaff2_resp.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Optimize
optimizer = DihedralOptimizer(max_terms=4, work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)

# --- Print parameters ---
print(f"\nRMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# --- Plot: Optimized ---
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.0, label="GAFF2 RESP (original)", alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label="Optimized")
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Psi4 RESP + Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile_resp.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Validate
validator = ParameterValidator(workDir)
val = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"\nQuality: {val['quality']} (RMSE: {val['rmse']:.4f} kcal/mol)")
```

!!! note "Psi4 is optional"
    This example requires `conda install -c conda-forge psi4 resp`.
    If Psi4 is not installed, omit `resp_charges_file` and use the default AM1-BCC charges:
    `amber_files = topo.generate_amber(conf['mol_file'])  # uses AM1-BCC by default`
