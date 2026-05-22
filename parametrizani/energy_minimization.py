"""
ParametrizANI - Energy Minimization
=====================================

OpenMM-based energy minimization with dihedral restraints.
Supports GAFF2 and OpenFF force fields.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from .utils import create_work_dir, relative_energies, write_energy_file

logger = logging.getLogger(__name__)


class EnergyMinimizer:
    """
    Perform OpenMM energy minimization with dihedral restraints.
    
    Parameters
    ----------
    force_field : str
        Force field to use ('gaff2', 'openff-2.0.0', etc.).
    work_dir : str
        Working directory for output files.
    """
    
    def __init__(self, force_field: str = 'gaff2', work_dir: str = './work'):
        self.force_field = force_field
        self.work_dir = create_work_dir(work_dir)
    
    def minimize_scan(
        self, topology_file: str, coordinate_file: str,
        conformer_files: List[str], dihedral_indices: List[int],
        angles: Optional[List[float]] = None, zero_dihedral: bool = True,
        force_constant: float = 1000.0, opt_tol: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Minimize conformers from a dihedral scan using OpenMM with AMBER topology.
        
        Parameters
        ----------
        topology_file : str
            Path to AMBER prmtop file.
        coordinate_file : str
            Path to AMBER inpcrd file.
        conformer_files : List[str]
            List of PDB files for each dihedral angle.
        dihedral_indices : List[int]
            Four atom indices defining the dihedral.
        zero_dihedral : bool
            If True, zero ALL torsion terms sharing the central bond (at2-at3).
            This matches the original ParametrizANI notebook behavior where all
            torsions around the central bond are zeroed, not just the specific quartet.
        force_constant : float
            Force constant for dihedral restraint.
        opt_tol : float
            Convergence tolerance for minimization (kJ/mol/nm).
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'angles', 'energies_relative', 'output_file'.
        """
        import openmm as mm
        from openmm import unit
        from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PDBFile, HBonds
        import math
        
        at1, at2, at3, at4 = dihedral_indices
        prmtop = AmberPrmtopFile(topology_file)
        inpcrd = AmberInpcrdFile(coordinate_file)
        
        if angles is None:
            angles = [float(os.path.splitext(os.path.basename(f))[0].split('.')[0]) for f in conformer_files]
        
        potential_energies = []
        logger.info(f"Minimizing {len(conformer_files)} conformers with OpenMM...")
        
        for i, (deg, pdb_file) in enumerate(zip(angles, conformer_files)):
            pdbfile = PDBFile(pdb_file)
            system = prmtop.createSystem(nonbondedCutoff=1*unit.nanometer, constraints=HBonds)
            
            for fi, force in enumerate(system.getForces()):
                force.setForceGroup(fi)
            
            if zero_dihedral:
                # Zero ALL torsion terms sharing the central bond (at2-at3).
                # This matches the original ParametrizANI notebook which checks only
                # the central pair atoms, ensuring consistency with the FRCMOD
                # modification that zeroes all central-pair type matches.
                for force in system.getForces():
                    if isinstance(force, mm.PeriodicTorsionForce):
                        for j in range(force.getNumTorsions()):
                            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(j)
                            if (p2 == at2 and p3 == at3) or (p2 == at3 and p3 == at2):
                                force.setTorsionParameters(
                                    j, p1, p2, p3, p4, 1, 0.0, 0.0
                                )
            
            energy_expression = "k*(1-cos(periodicity*theta-theta0))"
            custom_torsion = mm.CustomTorsionForce(energy_expression)
            custom_torsion.addGlobalParameter("k", force_constant)
            custom_torsion.addGlobalParameter("periodicity", 1)
            custom_torsion.addGlobalParameter("theta0", math.radians(deg))
            custom_torsion.addTorsion(at1, at2, at3, at4, [])
            system.addForce(custom_torsion)
            
            integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            simulation = mm.app.Simulation(prmtop.topology, system, integrator)
            simulation.context.setPositions(pdbfile.positions)
            simulation.minimizeEnergy(
                tolerance=opt_tol*unit.kilojoule_per_mole/unit.nanometer,
                maxIterations=1000000
            )
            
            state = simulation.context.getState(getEnergy=True)
            energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            potential_energies.append(energy_kj)
            
            min_pdb = os.path.join(self.work_dir, f'mol_files/{int(deg)}_min.pdb')
            os.makedirs(os.path.dirname(min_pdb), exist_ok=True)
            positions = simulation.context.getState(getPositions=True).getPositions()
            with open(min_pdb, 'w') as f:
                PDBFile.writeFile(simulation.topology, positions, f)
        
        energies_arr = np.array(potential_energies)
        energies_relative = relative_energies(energies_arr)
        suffix = "zeroed" if zero_dihedral else "default"
        output_file = os.path.join(self.work_dir, f'mm_energies_{suffix}.dat')
        write_energy_file(output_file, angles, energies_relative.tolist())
        
        logger.info(f"Minimization complete. MM energy range: {energies_relative.max():.3f} kcal/mol")
        
        return {
            'angles': angles,
            'energies': potential_energies,
            'energies_relative': energies_relative.tolist(),
            'output_file': output_file,
        }
    
    def minimize_scan_openff(
        self, smiles: str, conformer_files: List[str],
        dihedral_indices: List[int], angles: Optional[List[float]] = None,
        zero_dihedral: bool = True, force_constant: float = 1000.0, opt_tol: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Minimize conformers using OpenFF force field.
        
        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        conformer_files : List[str]
            List of PDB/MOL files.
        dihedral_indices : List[int]
            Four atom indices.
        zero_dihedral : bool
            If True, zero ALL torsion terms sharing the central bond (at2-at3).
        """
        import openmm as mm
        from openmm import unit
        from openmm.app import PDBFile, HBonds
        from openff.toolkit.topology import Molecule, Topology
        from openff.toolkit.typing.engines.smirnoff import ForceField
        import math
        
        at1, at2, at3, at4 = dihedral_indices
        molecule = Molecule.from_smiles(smiles)
        ff = ForceField(f'{self.force_field}.offxml')
        
        if angles is None:
            angles = [float(os.path.splitext(os.path.basename(f))[0].split('.')[0]) for f in conformer_files]
        
        potential_energies = []
        for deg, pdb_file in zip(angles, conformer_files):
            pdbfile = PDBFile(pdb_file)
            off_topology = Topology.from_openmm(pdbfile.topology, unique_molecules=[molecule])
            system = ff.create_openmm_system(off_topology)
            
            if zero_dihedral:
                # Zero ALL torsion terms sharing the central bond (at2-at3)
                for force in system.getForces():
                    if isinstance(force, mm.PeriodicTorsionForce):
                        for j in range(force.getNumTorsions()):
                            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(j)
                            if (p2 == at2 and p3 == at3) or (p2 == at3 and p3 == at2):
                                force.setTorsionParameters(
                                    j, p1, p2, p3, p4, 1, 0.0, 0.0
                                )
            
            energy_expression = "k*(1-cos(periodicity*theta-theta0))"
            custom_torsion = mm.CustomTorsionForce(energy_expression)
            custom_torsion.addGlobalParameter("k", force_constant)
            custom_torsion.addGlobalParameter("periodicity", 1)
            custom_torsion.addGlobalParameter("theta0", math.radians(deg))
            custom_torsion.addTorsion(at1, at2, at3, at4, [])
            system.addForce(custom_torsion)
            
            integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            simulation = mm.app.Simulation(pdbfile.topology, system, integrator)
            simulation.context.setPositions(pdbfile.positions)
            simulation.minimizeEnergy(
                tolerance=opt_tol*unit.kilojoule_per_mole/unit.nanometer,
                maxIterations=1000000
            )
            
            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            potential_energies.append(energy)
        
        energies_arr = np.array(potential_energies)
        energies_relative = relative_energies(energies_arr)
        output_file = os.path.join(self.work_dir, 'openff_mm_energies.dat')
        write_energy_file(output_file, angles, energies_relative.tolist())
        
        return {
            'angles': angles,
            'energies': potential_energies,
            'energies_relative': energies_relative.tolist(),
            'output_file': output_file,
        }
