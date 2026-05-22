"""ParametrizANI - Full Workflow

Complete pipeline from SMILES to topology files with verification.
Requires: pip install -e ".[full]"
          conda install -c conda-forge ambertools openmm openmmforcefields
          git clone https://github.com/pablo-arantes/AIMNet2.git  (optional, for AIMNet2)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
    get_dihedral_atom_types,
)


def main():
    smiles = 'C=CC(=O)OC'  # Vinyl acetate
    dihedral_indices = [0, 1, 2, 3]
    work_dir = './full_workflow_output'

    # User-configurable parameters
    model_name = 'torchani'  # ML model for reference energies
    opt_tol = 0.001          # Convergence threshold for geometry optimization (fmax)
    max_terms = 4            # Number of Fourier terms
    IDIVF = 1                # Scaling factor for equivalent torsions
    Force_constant = 1000    # Force constant for dihedral restraint (kcal/mol/rad²)
    mm_opt_tol = 0.001       # Convergence tolerance for MM minimization (kJ/mol/nm)
    step = 30                # Scan step size in degrees

    # Output directory for plots
    plot_dir = os.path.join(work_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # =========================================================================
    # 1. Generate conformer & scan
    # =========================================================================
    gen = ConformerGenerator(smiles, 'smiles', work_dir)
    conf = gen.run()
    scan = gen.generate_dihedral_conformers(dihedral_indices, step=step)
    print(f"[1/10] Generated {len(scan['angles'])} conformers")

    # =========================================================================
    # 2. Generate initial topology & detect atom types
    # =========================================================================
    topo = TopologyGenerator(work_dir, force_field='gaff2')
    amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')
    # Alternative: Use Psi4 RESP charges (requires psi4 + resp packages):
    #   from parametrizani import calculate_resp_charges
    #   resp = calculate_resp_charges(conf['mol_file'], method='HF', basis='6-31G*')
    #   amber_files = topo.generate_amber(conf['mol_file'], resp_charges_file=resp['output_file'])
    atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
    print(f"[2/10] Atom types detected: {atom_types}")

    # =========================================================================
    # 3. Calculate reference energies (geometry optimized with dihedral constrained)
    # =========================================================================
    calc = ReferenceEnergyCalculator(model_name, work_dir)
    ref = calc.scan_dihedral(
        scan['conformers'], scan['angles'],
        optimize=True, fmax=opt_tol,
        dihedral_indices=dihedral_indices
    )
    print(f"[3/10] Reference energy range: {max(ref['energies_relative']):.3f} kcal/mol")

    # --- Plot: Reference Energy Profile ---
    min_deg = min(ref['angles'])
    max_deg = max(ref['angles'])
    plt.figure(figsize=(8, 5))
    plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
    plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
    plt.xlabel('Dihedral Angle (degrees)')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.legend(frameon=True)
    plt.title('Reference Energy Profile')
    plt.savefig(os.path.join(plot_dir, f'{model_name}_reference.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       Saved: {plot_dir}/{model_name}_reference.png")

    # =========================================================================
    # 4. MM minimization with dihedral zeroed (isolates torsion contribution)
    # =========================================================================
    minimizer = EnergyMinimizer('gaff2', work_dir)
    mm = minimizer.minimize_scan(
        amber_files['prmtop'], amber_files['inpcrd'],
        scan['pdb_files'], dihedral_indices,
        angles=scan['angles'], zero_dihedral=True,
        force_constant=Force_constant, opt_tol=mm_opt_tol,
    )
    print(f"[4/10] MM energy range (dihedral zeroed): {max(mm['energies_relative']):.3f} kcal/mol")

    # --- Plot: Reference vs MM ---
    plt.figure(figsize=(8, 5))
    plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
    plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label="GAFF2")
    plt.xticks(np.arange(min_deg, max_deg + 1, 60.0))
    plt.xlabel('Dihedral Angle (degrees)')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.legend(frameon=True)
    plt.title('Reference vs GAFF2 Energy Profile')
    plt.savefig(os.path.join(plot_dir, f'{model_name}_vs_gaff2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       Saved: {plot_dir}/{model_name}_vs_gaff2.png")

    # =========================================================================
    # 5. Optimize dihedral parameters
    # =========================================================================
    optimizer = DihedralOptimizer(max_terms=max_terms, work_dir=work_dir)
    result = optimizer.run_optimization(
        ref['angles'], ref['energies_relative'],
        mm_energies=mm['energies_relative'],
        atom_types=atom_types
    )
    print(f"[5/10] Best RMSE ({max_terms} terms): {result['rmse']:.4f} kcal/mol")

    # --- Print RMSE per number of terms ---
    print(f"\n       RMSE per number of terms:")
    for i, rmse in enumerate(result['all_rmse'], 1):
        print(f"         {i} terms: {rmse:.4f} kcal/mol")

    # --- Print optimized parameters ---
    print(f"\n       Optimized FRCMOD Parameters:")
    print(f"       {result['frcmod_params']}")

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
    plt.savefig(os.path.join(plot_dir, 'optimized_profile.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       Saved: {plot_dir}/optimized_profile.png")

    # =========================================================================
    # 6. Validate fit quality
    # =========================================================================
    validator = ParameterValidator(work_dir)
    val = validator.validate_parameters(
        ref['angles'], ref['energies_relative'], result['best_fit']
    )
    print(f"\n[6/10] Quality: {val['quality']} (R\u00b2: {val['r_squared']:.4f})")

    # =========================================================================
    # 7. Write FRCMOD and update with optimized parameters
    # =========================================================================
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
    print(f"[7/10] Updated FRCMOD: {updated_frcmod}")

    # =========================================================================
    # 8. Generate final topology with optimized parameters
    # =========================================================================
    new_amber = topo.generate_amber(
        conf['mol_file'],
        frcmod_file=updated_frcmod,
        mol2_file=topology_mol2,
        output_prefix='SYS_new',
    )
    print(f"[8/10] New AMBER topology: {new_amber['prmtop']}")

    # =========================================================================
    # 9. Export to GROMACS and OpenMM formats
    # =========================================================================
    try:
        gromacs_files = topo.generate_gromacs(new_amber['prmtop'], new_amber['inpcrd'])
        print(f"[9/10] GROMACS files: {gromacs_files['top']}")
    except Exception as e:
        print(f"[9/10] GROMACS export skipped: {e}")

    try:
        openmm_files = topo.generate_openmm(new_amber['prmtop'], new_amber['inpcrd'])
        print(f"       OpenMM files: {openmm_files['xml']}")
    except Exception as e:
        print(f"       OpenMM export skipped: {e}")

    # =========================================================================
    # 10. Verify: re-run MM scan with new topology (zero_dihedral=False)
    # =========================================================================
    try:
        mm_new = minimizer.minimize_scan(
            new_amber['prmtop'], new_amber['inpcrd'],
            scan['pdb_files'], dihedral_indices,
            angles=scan['angles'], zero_dihedral=False,
            force_constant=Force_constant, opt_tol=mm_opt_tol,
        )
        rmse_verify = np.sqrt(np.mean(
            (np.array(ref['energies_relative']) - np.array(mm_new['energies_relative']))**2
        ))
        print(f"[10/10] Verification RMSE (Ref vs New Topology): {rmse_verify:.4f} kcal/mol")
        if rmse_verify < 1.0:
            print("        \u2705 Parameters validated successfully!")
        else:
            print("        \u26a0\ufe0f RMSE > 1.0 - consider adjusting n_terms or IDIVF")

        # --- Plot: Verification (Reference vs Original vs Optimized Topology) ---
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
        plt.savefig(os.path.join(plot_dir, 'verification_final.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"        Saved: {plot_dir}/verification_final.png")

    except Exception as e:
        print(f"[10/10] Verification skipped: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Done! Results in: {work_dir}")
    print(f"{'='*60}")
    print(f"\nPlots saved in: {plot_dir}/")
    print(f"  - {model_name}_reference.png")
    print(f"  - {model_name}_vs_gaff2.png")
    print(f"  - optimized_profile.png")
    print(f"  - verification_final.png")
    print(f"\nOptimized parameters (FRCMOD format):")
    print(selected_frcmod_params)


if __name__ == '__main__':
    main()
