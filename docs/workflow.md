# Workflow

## Pipeline Overview

```
SMILES/PDB/MOL
     ↓
1. Conformer Generation (RDKit + MMFF94)
     ↓
2. Dihedral Scan (constrained rotation)
     ↓
3. Topology Generation & Atom Type Detection (antechamber + tLEaP)
     ↓
4. Reference Energy Calculation (TorchANI/MACE/xTB/AIMNet2, constrained optimization)
     ↓
5. MM Energy Minimization (OpenMM, dihedral zeroed around central bond)
     ↓
6. Dihedral Optimization (Least-squares fitting, 1–6 Fourier terms)
     ↓
7. Validation (RMSE, MAE, R², quality rating)
     ↓
8. Write FRCMOD & Update Topology (zero central-pair, delete exact, insert optimized)
     ↓
9. Generate Final Topology (AMBER/GROMACS/OpenMM with new parameters)
     ↓
10. Verification (MM scan with new topology, zero_dihedral=False)
```

---

## Step 1: Conformer Generation

Generates a 3D conformer from SMILES/PDB/MOL using RDKit with MMFF94 optimization.

```python
gen = ConformerGenerator(smiles, 'smiles', workDir)
conf = gen.run()
```

**Output:** Optimized 3D structure (`conf['mol_file']`, `conf['pdb_file']`)

## Step 2: Dihedral Scan

Rotates the specified dihedral in fixed increments, generating conformers at each angle.
Uses `rdMolTransforms.SetDihedralDeg()` followed by MMFF optimization (with the dihedral constrained).

```python
scan = gen.generate_dihedral_conformers(dihedral_indices, step=30)
```

**Parameters:**

- `dihedral_indices` — 0-based indices of the 4 atoms defining the rotatable bond
- `step` — Rotation increment in degrees (default: 30°)

**Output:** List of conformer files and angles (`scan['conformers']`, `scan['angles']`)

## Step 3: Topology Generation

Generates AMBER topology using antechamber (for charges) and tLEaP (for parameters).
Automatically detects atom types from the MOL2 file.

```python
topo = TopologyGenerator(workDir, force_field='gaff2')

# Option A: AM1-BCC charges (default, fast)
amber_files = topo.generate_amber(conf['mol_file'])

# Option B: Use Psi4 RESP charges (higher accuracy)
from parametrizani import calculate_resp_charges
resp = calculate_resp_charges(conf['mol_file'], method='HF', basis='6-31G*')
amber_files = topo.generate_amber(conf['mol_file'], resp_charges_file=resp['output_file'])

atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
```

The `resp_charges_file` parameter tells antechamber to read pre-computed charges
(`-c rc -cf <file>`) instead of computing them internally, avoiding the need for
Gaussian/GAMESS that antechamber's built-in `-c resp` mode requires.

**Output:** `.prmtop`, `.crd`, `.mol2`, `.frcmod`, `.lib` files

## Step 4: Reference Energy Calculation

Calculates the energy at each scan angle using an ML potential.
Geometry is optimized with the dihedral angle constrained (via ASE `FixInternals`).

```python
calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(
    scan['conformers'], scan['angles'],
    optimize=True, fmax=0.001,
    dihedral_indices=dihedral_indices
)
```

**Key parameters:**

- `fmax` — Force convergence threshold for ASE's LBFGS optimizer (smaller = tighter geometry optimization)
- `dihedral_indices` — Constrains the dihedral at its current value during optimization

**Plot: Reference Energy Profile**

```python
import numpy as np
import matplotlib.pyplot as plt

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
```

## Step 5: MM Energy Minimization

Minimizes each conformer with OpenMM using the AMBER topology, with ALL torsion terms
around the central bond zeroed. This isolates the non-torsional contribution for fitting.

```python
minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True,
    force_constant=1000, opt_tol=0.001,
)
```

!!! important "Central bond zeroing"
    When `zero_dihedral=True`, ALL torsion terms sharing the central bond (atom2–atom3)
    are zeroed, not just the specific 4-atom quartet. This matches the original
    ParametrizANI notebook behavior and ensures consistency with the FRCMOD modification.

**Plot: Reference vs MM**

```python
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
```

## Step 6: Dihedral Optimization

Fits Fourier series coefficients to minimize the difference between reference and MM energies.
Uses linear least-squares regression.

```python
optimizer = DihedralOptimizer(max_terms=6, work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)
```

The optimizer tries 1 through `max_terms` Fourier terms and reports RMSE for each.

**Print RMSE and optimized parameters:**

```python
print(f"\nRMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])
```

**Plot: Optimized Profile**

```python
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
```

## Step 7: Validation

Evaluates the fit quality with standard metrics.

```python
validator = ParameterValidator(workDir)
validation = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"Quality: {validation['quality']} (RMSE: {validation['rmse']:.4f} kcal/mol)")
```

**Quality ratings:**

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | ≤ 0.25 | Near-QM accuracy |
| Good | ≤ 0.50 | Suitable for most applications |
| Acceptable | ≤ 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

## Step 8: Write FRCMOD & Update Topology

Writes optimized parameters to FRCMOD format and updates the original FRCMOD file:

1. **Zero** all DIHE lines matching the central 2-atom pattern (e.g., `ca-os`)
2. **Delete** DIHE lines matching the exact 4-atom pattern (forward or reversed)
3. **Add explicit zeros** for all other quartets around the central bond (blocks generic wildcard fallback from gaff2.dat)
4. **Insert** new optimized lines after the DIHE header

```python
selected_frcmod_params = optimizer.format_frcmod_params(
    result, n_terms=max_terms, idivf=IDIVF
)
updated_frcmod = topo.update_frcmod(
    amber_files['frcmod'],
    selected_frcmod_params,
    dihedral_indices=dihedral_indices,
    mol2_file=topology_mol2,
)
```

**Key parameters:**

- `n_terms` — Number of Fourier terms to write (1–6)
- `idivf` — IDIVF scaling factor for equivalent torsions
- `dihedral_indices` + `mol2_file` — Used to derive atom-type patterns from the actual MOL2

!!! important "Generic wildcard override"
    `update_frcmod` now generates explicit zero lines for ALL dihedral quartets around
    the central bond, even those not present in the original FRCMOD. This prevents
    tLEaP from falling back to generic wildcard parameters (e.g., `X-ce-c-X` from
    gaff2.dat), which would add unwanted torsion energy on top of the optimized parameters.

## Step 9: Generate Final Topology

Regenerates the topology with the updated FRCMOD using two-stage tLEaP.

```python
new_amber = topo.generate_amber(
    conf['mol_file'],
    frcmod_file=updated_frcmod,
    mol2_file=topology_mol2,
    output_prefix='SYS_new',
)

# Export to other formats
gromacs_files = topo.generate_gromacs(new_amber['prmtop'], new_amber['inpcrd'])
openmm_files = topo.generate_openmm(new_amber['prmtop'], new_amber['inpcrd'])
```

!!! note "GROMACS and OpenMM export"
    Both `generate_gromacs()` and `generate_openmm()` use `parmed.openmm.load_topology()`
    to create the output, ensuring all optimized dihedral parameters are correctly
    transferred to the target format.

## Step 10: Verification

Re-runs MM minimization with the new topology **without** zeroing the dihedral.
The resulting energy profile should match the reference (fitted) curve.

```python
mm_new = minimizer.minimize_scan(
    new_amber['prmtop'], new_amber['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=False,
)
```

**Plot: Verification (Reference vs Original vs Optimized Topology)**

```python
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
```

**Print verification RMSE:**

```python
rmse_new = np.sqrt(np.mean(
    (np.array(ref['energies_relative']) - np.array(mm_new['energies_relative']))**2
))
print(f"\nRMSE (Reference vs New Topology): {rmse_new:.4f} kcal/mol")
if rmse_new < 1.0:
    print("  Parameters validated successfully!")
else:
    print("  RMSE > 1.0 - consider adjusting the number of Fourier terms")
```

If the verification RMSE is < 1.0 kcal/mol, the parameters are validated.
