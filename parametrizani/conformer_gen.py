"""
ParametrizANI - Conformer Generator
====================================

Generate 3D conformers from SMILES strings, PDB files, or MOL files.
Uses RDKit for molecule handling and MMFF94 for initial optimization.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from .utils import create_work_dir, extract_atom_info_from_pdb

logger = logging.getLogger(__name__)


class ConformerGenerator:
    """
    Generate 3D molecular conformers from SMILES, PDB, or MOL input.
    
    Uses RDKit for molecule handling and MMFF94 force field for 
    initial geometry optimization.
    
    Parameters
    ----------
    molecule_input : str
        SMILES string or path to PDB/MOL file.
    input_type : str
        Type of input: 'smiles', 'pdb', or 'mol'.
    work_dir : str
        Working directory for output files.
    
    Examples
    --------
    >>> from parametrizani import ConformerGenerator
    >>> gen = ConformerGenerator('CC(=O)OC', 'smiles', './work')
    >>> result = gen.run()
    >>> print(result['pdb_file'])
    """
    
    def __init__(self, molecule_input: str, input_type: str = 'smiles', 
                 work_dir: str = './work'):
        self.molecule_input = molecule_input
        self.input_type = input_type.lower()
        self.work_dir = create_work_dir(work_dir)
        self.mol = None
        self.hmol = None  # Molecule with hydrogens
        
        if self.input_type not in ['smiles', 'pdb', 'mol']:
            raise ValueError(f"Invalid input type: {self.input_type}. "
                           f"Must be 'smiles', 'pdb', or 'mol'.")
    
    def run(self, optimize: bool = True, max_iters: int = 200) -> Dict[str, Any]:
        """
        Generate and optimize a 3D conformer.
        
        Parameters
        ----------
        optimize : bool
            Whether to optimize with MMFF94. Default True.
        max_iters : int
            Maximum optimization iterations. Default 200.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'mol': RDKit molecule object
            - 'pdb_file': Path to output PDB file
            - 'mol_file': Path to output MOL file
            - 'smiles': Canonical SMILES
            - 'num_atoms': Number of atoms
            - 'atom_info': List of atom information dicts
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdForceFieldHelpers
        
        # Load molecule based on input type
        if self.input_type == 'smiles':
            self.mol = Chem.MolFromSmiles(self.molecule_input)
            if self.mol is None:
                raise ValueError(f"Invalid SMILES: {self.molecule_input}")
            self.hmol = Chem.AddHs(self.mol)
            AllChem.EmbedMolecule(self.hmol, randomSeed=42)
            
        elif self.input_type == 'pdb':
            self.hmol = Chem.MolFromPDBFile(self.molecule_input, removeHs=False)
            if self.hmol is None:
                raise ValueError(f"Could not read PDB file: {self.molecule_input}")
            self.mol = Chem.RemoveHs(self.hmol)
            
        elif self.input_type == 'mol':
            self.hmol = AllChem.MolFromMolFile(self.molecule_input, removeHs=False)
            if self.hmol is None:
                raise ValueError(f"Could not read MOL file: {self.molecule_input}")
            self.mol = Chem.RemoveHs(self.hmol)
        
        # Optimize with MMFF94
        if optimize:
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.hmol)
            if mp is not None:
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.hmol, mp)
                if ff is not None:
                    AllChem.OptimizeMolecule(ff, maxIters=max_iters)
                    ff.Minimize(maxIts=1000)
                    logger.info("MMFF94 optimization completed.")
                else:
                    logger.warning("Could not create MMFF force field, skipping optimization.")
            else:
                logger.warning("Could not get MMFF properties, skipping optimization.")
        
        # Save output files
        pdb_file = os.path.join(self.work_dir, 'molecule.pdb')
        mol_file = os.path.join(self.work_dir, 'molecule.mol')
        
        AllChem.MolToPDBFile(self.hmol, pdb_file)
        AllChem.MolToMolFile(self.hmol, mol_file)
        
        # Get canonical SMILES
        smiles = Chem.MolToSmiles(self.mol)
        
        # Extract atom information
        atom_info = extract_atom_info_from_pdb(pdb_file)
        
        logger.info(f"Conformer generated: {smiles} ({self.hmol.GetNumAtoms()} atoms)")
        
        return {
            'mol': self.hmol,
            'mol_no_h': self.mol,
            'pdb_file': pdb_file,
            'mol_file': mol_file,
            'smiles': smiles,
            'num_atoms': self.hmol.GetNumAtoms(),
            'atom_info': atom_info,
        }
    
    def generate_dihedral_conformers(
        self,
        dihedral_indices: List[int],
        min_angle: int = -180,
        max_angle: int = 180,
        step: int = 15,
        constraint_force: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Generate conformers by scanning a dihedral angle.
        
        Uses rdMolTransforms.SetDihedralDeg to explicitly set the dihedral geometry,
        then optimizes with MMFF94 while constraining the dihedral and atomic positions.
        
        Parameters
        ----------
        dihedral_indices : List[int]
            Four atom indices defining the dihedral.
        min_angle : int
            Minimum dihedral angle in degrees. Default -180.
        max_angle : int
            Maximum dihedral angle in degrees. Default 180.
        step : int
            Step size in degrees. Default 15.
        constraint_force : float
            Force constant for the torsion constraint. Default 1000.0.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'angles': List of dihedral angles
            - 'conformers': List of MOL file paths
            - 'pdb_files': List of PDB file paths
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolTransforms
        from rdkit import rdBase
        rdBase.BlockLogs()
        
        if self.hmol is None:
            raise RuntimeError("Must call run() before generating dihedral conformers.")
        
        at1, at2, at3, at4 = dihedral_indices
        
        mol_dir = os.path.join(self.work_dir, 'mol_files')
        os.makedirs(mol_dir, exist_ok=True)
        
        angles = []
        mol_files = []
        pdb_files = []
        
        for deg in range(min_angle, max_angle + step, step):
            # KEY: Explicitly set the dihedral angle geometry on the molecule
            rdMolTransforms.SetDihedralDeg(
                self.hmol.GetConformer(0),
                int(at1), int(at2), int(at3), int(at4),
                float(deg)
            )
            
            # Get MMFF properties and force field
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.hmol)
            if mp is None:
                logger.warning(f"Could not get MMFF properties for angle {deg}")
                continue
            
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.hmol, mp)
            if ff is None:
                logger.warning(f"Could not create MMFF force field for angle {deg}")
                continue
            
            # Add torsion constraint to maintain the dihedral angle
            ff.MMFFAddTorsionConstraint(
                int(at1), int(at2), int(at3), int(at4),
                False, float(deg), float(deg), constraint_force
            )
            
            # Add position constraints to all atoms
            new_match = self.hmol.GetSubstructMatch(self.hmol)
            for atidx in new_match:
                ff.MMFFAddPositionConstraint(atidx, 0.05, 200)
            
            # Minimize with constrained optimization
            max_iters = 10
            while ff.Minimize(maxIts=1000) and max_iters > 0:
                max_iters -= 1
            
            # Save files
            mol_path = os.path.join(mol_dir, f'{deg}.mol')
            pdb_path = os.path.join(mol_dir, f'{deg}.pdb')
            
            AllChem.MolToMolFile(self.hmol, mol_path)
            AllChem.MolToPDBFile(self.hmol, pdb_path)
            
            angles.append(deg)
            mol_files.append(mol_path)
            pdb_files.append(pdb_path)
        
        logger.info(f"Generated {len(angles)} dihedral conformers "
                   f"({min_angle}\u00b0 to {max_angle}\u00b0, step {step}\u00b0)")
        
        return {
            'angles': angles,
            'conformers': mol_files,
            'pdb_files': pdb_files,
        }
