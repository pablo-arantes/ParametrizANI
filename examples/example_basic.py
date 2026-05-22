"""ParametrizANI - Basic Example

Demonstrates dihedral optimization using pre-computed energy files.
This is the simplest usage: just provide reference and MM energy profiles.

The .dat files contain two columns: angle (degrees) and energy (kcal/mol).
"""
from parametrizani import optimize_dihedral, validate_parameters, read_energy_file
from parametrizani import DihedralOptimizer


def main():
    # Read pre-computed energy profiles
    ref_angles, ref_energies = read_energy_file('examples/qm.dat')
    mm_angles, mm_energies = read_energy_file('examples/mm.dat')

    # User-configurable parameters
    max_terms = 6   # Number of Fourier terms (1-6)
    IDIVF = 1       # Scaling factor for equivalent torsions
    atom_types = ['ca', 'ca', 'ca', 'ca']  # 4-atom dihedral type pattern

    # Optimize dihedral parameters
    result = optimize_dihedral(
        ref_angles, ref_energies,
        atom_types=atom_types,
        mm_energies=mm_energies,
        max_terms=max_terms,
        work_dir='./output'
    )

    print(f"Best RMSE ({len(result['best_parameters'])} terms): {result['rmse']:.4f} kcal/mol")
    print(f"\nFRCMOD Parameters (all {max_terms} terms):")
    print(result['frcmod_params'])

    # You can also select a specific number of terms and IDIVF:
    optimizer = DihedralOptimizer(max_terms=max_terms, work_dir='./output')
    selected_params = optimizer.format_frcmod_params(
        result, n_terms=3, idivf=IDIVF  # Use only 3 Fourier terms
    )
    print(f"\nFRCMOD Parameters (3 terms, IDIVF={IDIVF}):")
    print(selected_params)

    # Validate
    val = validate_parameters(ref_angles, ref_energies, result['best_fit'], './output')
    print(f"\nQuality: {val['quality']} (R\u00b2: {val['r_squared']:.4f})")


if __name__ == '__main__':
    main()
