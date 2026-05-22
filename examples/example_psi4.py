"""ParametrizANI - Psi4 RESP Charges Example

Geometry optimization + RESP charge calculation + GAFF2 topology generation.
Requires: pip install parametrizani
          conda install -c conda-forge psi4 resp ambertools
"""
import os
from parametrizani import (
    ConformerGenerator,
    TopologyGenerator,
    calculate_resp_charges,
)


def main():
    smiles = 'CC(=O)OC'  # Methyl acetate
    work_dir = './psi4_resp_output'

    # =========================================================================
    # 1. Generate conformer
    # =========================================================================
    gen = ConformerGenerator(smiles, 'smiles', work_dir)
    conf = gen.run()
    print(f"[1/3] Molecule: {conf['smiles']} ({conf['num_atoms']} atoms)")

    # =========================================================================
    # 2. Calculate RESP charges with Psi4
    # =========================================================================
    resp = calculate_resp_charges(
        conf['mol_file'],
        method='HF',           # Standard RESP protocol (Bayly et al. 1993)
        basis='6-31G*',        # Standard basis for RESP
        charge=0,              # Net molecular charge
        multiplicity=1,        # Singlet
        memory='2 GB',         # Psi4 memory allocation
        n_threads=2,           # CPU threads
    )

    print(f"[2/3] RESP charges calculated ({resp['method']}/{resp['basis']})")
    print(f"      Charges file: {resp['output_file']}")
    print(f"\n      {'Atom':<6} {'Charge':>10}")
    print(f"      {'-'*6} {'-'*10}")
    for symbol, q in zip(resp['atom_symbols'], resp['charges']):
        print(f"      {symbol:2s}    {q:10.6f}")
    print(f"\n      Total: {resp['charges'].sum():.6f}")

    # =========================================================================
    # 3. Generate GAFF2 topology using Psi4 RESP charges
    # =========================================================================
    topo = TopologyGenerator(work_dir, force_field='gaff2')
    amber_files = topo.generate_amber(
        conf['mol_file'],
        resp_charges_file=resp['output_file']  # Uses Psi4 charges via antechamber -c rc
    )

    print(f"\n[3/3] GAFF2 topology generated with RESP charges")
    print(f"      PRMTOP: {amber_files['prmtop']}")
    print(f"      INPCRD: {amber_files['inpcrd']}")
    print(f"      MOL2:   {amber_files['mol2']}")
    print(f"      FRCMOD: {amber_files['frcmod']}")

    # =========================================================================
    # (Optional) Export to GROMACS or OpenMM
    # =========================================================================
    try:
        gmx = topo.generate_gromacs(amber_files['prmtop'], amber_files['inpcrd'])
        print(f"\n      GROMACS: {gmx['top']}")
    except Exception as e:
        print(f"\n      GROMACS export skipped: {e}")

    try:
        omm = topo.generate_openmm(amber_files['prmtop'], amber_files['inpcrd'])
        print(f"      OpenMM:  {omm['xml']}")
    except Exception as e:
        print(f"      OpenMM export skipped: {e}")

    print("\n✓ Done! Topology files ready with Psi4 RESP charges.")


if __name__ == '__main__':
    main()
