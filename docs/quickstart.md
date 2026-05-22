# Quick Start

## Basic Usage (Pre-computed Energies)

If you already have reference and MM energy profiles (e.g., from Gaussian or external QM):

```python
from parametrizani import optimize_dihedral, validate_parameters, read_energy_file
from parametrizani import DihedralOptimizer
import numpy as np
import matplotlib.pyplot as plt

# Read pre-computed energy files (angle  energy format)
ref_angles, ref_energies = read_energy_file('qm.dat')
mm_angles, mm_energies = read_energy_file('mm.dat')

# Optimize dihedral parameters
result = optimize_dihedral(
    ref_angles, ref_energies,
    atom_types=['ca', 'ca', 'os', 'ca'],
    mm_energies=mm_energies,
    max_terms=6,
    work_dir='./output'
)

# --- Print RMSE and parameters ---
print(f"RMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# Select specific number of terms and IDIVF
optimizer = DihedralOptimizer(max_terms=6, work_dir='./output')
selected = optimizer.format_frcmod_params(result, n_terms=3, idivf=1)
print(f"\n3-term FRCMOD Parameters:")
print(selected)

# --- Plot: Optimized Profile ---
plt.figure(figsize=(8, 5))
plt.plot(ref_angles, ref_energies, 'o-', linewidth=1.5, label='Reference (QM)')
plt.plot(mm_angles, mm_energies, 's--', linewidth=1.0, label='MM (dihedral zeroed)', alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label='Optimized')
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Full Workflow (GAFF2)

Complete pipeline from SMILES to optimized topology:

```python
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
    get_dihedral_atom_types,
)
import os
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
smiles = 'COc3ccc2c(=O)cc(c1ccccc1)oc2c3'
dihedral_indices = [8, 9, 10, 15]
workDir = './work'

model_name = 'torchani'   # ML model for reference energies
opt_tol = 0.001           # Convergence threshold (fmax) for geometry optimization
max_terms = 6             # Number of Fourier terms
IDIVF = 1                 # Scaling factor for equivalent torsions
Force_constant = 1000     # Dihedral restraint force constant (kcal/mol/rad²)

# === Step 1: Generate conformer and dihedral scan ===
gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

# === Step 2: Generate initial topology & detect atom types ===
topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])  # Uses AM1-BCC charges by default
# For Psi4 RESP charges instead:
#   from parametrizani import calculate_resp_charges
#   resp = calculate_resp_charges(conf['mol_file'], method='HF', basis='6-31G*')
#   amber_files = topo.generate_amber(conf['mol_file'], resp_charges_file=resp['output_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
print(f"Atom types: {atom_types}")  # e.g. ['ca', 'ca', 'os', 'ca']

# === Step 3: Reference energy calculation ===
calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=opt_tol,
    dihedral_indices=dihedral_indices
)

# --- Plot: Reference Energy Profile ---
min_deg, max_deg = min(ref['angles']), max(ref['angles'])
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference Energy Profile')
plt.savefig(f'{model_name}_reference.png', dpi=300, bbox_inches='tight')
plt.show()

# === Step 4: MM minimization (dihedral zeroed) ===
minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True,
    force_constant=Force_constant, opt_tol=0.001,
)

# --- Plot: Reference vs MM ---
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

# === Step 5: Optimize dihedral parameters ===
optimizer = DihedralOptimizer(max_terms=max_terms, work_dir=workDir)
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

# === Step 6: Validate ===
validator = ParameterValidator(workDir)
validation = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"\nQuality: {validation['quality']} (RMSE: {validation['rmse']:.4f} kcal/mol)")

# === Step 7: Write FRCMOD & update topology ===
selected_frcmod_params = optimizer.format_frcmod_params(
    result, n_terms=max_terms, idivf=IDIVF
)
frcmod_file = optimizer.write_frcmod(result, idivf=IDIVF, n_terms=max_terms)

topology_mol2 = os.path.join(topo.tleap_dir, 'new.mol2')
if not os.path.exists(topology_mol2):
    topology_mol2 = amber_files['mol2']

updated_frcmod = topo.update_frcmod(
    amber_files['frcmod'],
    selected_frcmod_params,
    dihedral_indices=dihedral_indices,
    mol2_file=topology_mol2,
)

# === Step 8: Generate final topology ===
new_amber = topo.generate_amber(
    conf['mol_file'],
    frcmod_file=updated_frcmod,
    mol2_file=topology_mol2,
    output_prefix='SYS_new',
)

# === Step 9: Export to GROMACS / OpenMM ===
gromacs_files = topo.generate_gromacs(new_amber['prmtop'], new_amber['inpcrd'])
openmm_files = topo.generate_openmm(new_amber['prmtop'], new_amber['inpcrd'])

# === Step 10: Verify ===
mm_new = minimizer.minimize_scan(
    new_amber['prmtop'], new_amber['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=False,
    force_constant=Force_constant,
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

## Full Workflow (OpenFF)

For OpenFF force fields, no topology generation step is needed for MM minimization:

```python
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
)
import numpy as np
import matplotlib.pyplot as plt

smiles = 'COc3ccc2c(=O)cc(c1ccccc1)oc2c3'
dihedral_indices = [8, 9, 10, 15]
workDir = './work'
model_name = 'torchani'

# Steps 1 & 3: Conformer and reference (same as GAFF2)
gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)

calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=0.001,
    dihedral_indices=dihedral_indices
)

# Step 4: MM minimization with OpenFF (no topology needed)
minimizer = EnergyMinimizer('openff-2.0.0', workDir)
mm = minimizer.minimize_scan_openff(
    smiles,
    scan['pdb_files'], dihedral_indices,
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

# Steps 5-6: Optimize and validate (same as GAFF2)
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
validation = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"\nQuality: {validation['quality']} (RMSE: {validation['rmse']:.4f} kcal/mol)")
```
