"""Test macrocycle ring-dihedral scan support.

RDKit's ``rdMolTransforms.SetDihedralDeg`` rejects ring bonds because ring
closure couples them to their neighbors. The fix in conformer_gen.py
auto-detects ring bonds and routes them to a relaxation-only path
(MMFFAddTorsionConstraint + weakened ring-atom position constraints).

Cyclooctene (8-membered ring) is the smallest non-aromatic ring large enough
to support a meaningful dihedral scan. The C-C-C-C sp3 dihedral around the
double bond exercises the new ring-dihedral path.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from parametrizani import ConformerGenerator


def _find_ring_dihedral(mol):
    """Return four atom indices for a non-aromatic single-bond ring dihedral."""
    from rdkit import Chem
    ri = mol.GetRingInfo()
    macro = max(ri.AtomRings(), key=len)
    for i in range(len(macro)):
        b, c = macro[i], macro[(i + 1) % len(macro)]
        bond = mol.GetBondBetweenAtoms(b, c)
        if bond is None or bond.GetIsAromatic():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a = macro[(i - 1) % len(macro)]
        d = macro[(i + 2) % len(macro)]
        return [a, b, c, d]
    raise RuntimeError("no rotatable ring dihedral found")


def test_cyclooctene_ring_dihedral_scan_completes():
    """Ring-dihedral scan must complete without ValueError on macrocycle.

    Before the fix: SetDihedralDeg raised "bond (j,k) must not belong to a
    ring" and produced 0 conformers.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Z-cyclooctene — 8-ring with one cis double bond.
        gen = ConformerGenerator(r"C1=CCCCCCC1", "smiles", tmp)
        gen.run()
        mol = gen.hmol
        dihedral = _find_ring_dihedral(mol)

        result = gen.generate_dihedral_conformers(
            dihedral, min_angle=-180, max_angle=180, step=60
        )

        # We expect every requested angle to produce a conformer when the
        # ring path is enabled (relaxation handles the strain). Allow at most
        # a single missed angle to keep the test robust to RDKit version
        # differences.
        n_requested = (180 - (-180)) // 60 + 1
        assert len(result["angles"]) >= n_requested - 1, (
            f"only {len(result['angles'])} of {n_requested} scan points generated "
            f"on cyclooctene; before the fix this returned 0"
        )

        # All output files must exist
        for path in result["conformers"] + result["pdb_files"]:
            assert os.path.exists(path)


def test_open_chain_dihedral_unaffected():
    """Non-ring dihedrals must still use the original SetDihedralDeg path.

    Regression check — the new branch must not break the open-chain case.
    n-butane's central C-C is the canonical test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        gen = ConformerGenerator("CCCC", "smiles", tmp)
        gen.run()
        mol = gen.hmol
        # n-butane: heavy atoms 0-1-2-3 are the C-C-C-C dihedral. With Hs,
        # the carbon indices are still the first four after AddHs.
        carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        assert len(carbons) == 4
        dihedral = carbons  # C0-C1-C2-C3

        result = gen.generate_dihedral_conformers(
            dihedral, min_angle=-180, max_angle=180, step=60
        )

        n_requested = (180 - (-180)) // 60 + 1
        assert len(result["angles"]) == n_requested


if __name__ == "__main__":
    test_cyclooctene_ring_dihedral_scan_completes()
    test_open_chain_dihedral_unaffected()
    print("OK")
