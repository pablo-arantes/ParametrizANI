"""
ParametrizANI - Reference Energy Calculator
=============================================

Calculate reference energies using ML potentials:
- TorchANI (ANI-2x, ANI-1x, ANI-1ccx, ANI-2xr, ANI-2dr, ANI-2xr-Snn, ANI-mbis)
- AIMNet2 (wB97M-D3, B97-3c)
- MACE-OFF
- GFN2-xTB

All ANI models are loaded from the torchani package (https://github.com/aiqm/torchani).
AIMNet2 models are loaded from https://github.com/pablo-arantes/AIMNet2.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .utils import create_work_dir, relative_energies, write_energy_file, HARTREE_TO_KCAL

logger = logging.getLogger(__name__)


# =============================================================================
# AIMNet2 ASE Calculator (proper inheritance from ase.calculators.calculator.Calculator)
# =============================================================================
import ase.calculators.calculator


class AIMNet2Calculator(ase.calculators.calculator.Calculator):
    """
    ASE-compatible calculator for AIMNet2 models.
    
    Properly inherits from ase.calculators.calculator.Calculator so it works
    with ASE optimizers (LBFGS, BFGS, etc.) and constraints (FixInternals).
    
    Based on the original implementation from:
    https://github.com/pablo-arantes/AIMNet2
    """
    implemented_properties = ['energy', 'forces', 'free_energy', 'charges']

    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model
        self.charge = charge
        self.device = next(model.parameters()).device
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
        self.cutoff = float(cutoff)
        self._t_numbers = None
        self._t_charge = None

    def do_reset(self):
        self._t_numbers = None
        self._t_charge = None
        self.charge = 0.0

    def set_charge(self, charge):
        self.charge = float(charge)

    def _make_input(self):
        import torch
        coord = torch.as_tensor(
            self.atoms.positions
        ).to(torch.float).to(self.device).unsqueeze(0)
        if self._t_numbers is None:
            self._t_numbers = torch.as_tensor(
                self.atoms.numbers
            ).to(torch.long).to(self.device).unsqueeze(0)
            self._t_charge = torch.tensor(
                [self.charge], dtype=torch.float, device=self.device
            )
        d = dict(coord=coord, numbers=self._t_numbers, charge=self._t_charge)
        return d

    def _eval_model(self, d, forces=True):
        import torch
        prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(forces)
        if forces:
            d['coord'].requires_grad_(True)
        _out = self.model(d)
        ret = dict(
            energy=_out['energy'].item(),
            charges=_out['charges'].detach()[0].cpu().numpy()
        )
        if forces:
            if 'forces' in _out:
                f = _out['forces'][0]
            else:
                f = -torch.autograd.grad(_out['energy'], d['coord'])[0][0]
            ret['forces'] = f.detach().cpu().numpy()
        torch._C._set_grad_enabled(prev)
        return ret

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)
        _in = self._make_input()
        do_forces = 'forces' in properties
        _out = self._eval_model(_in, do_forces)
        self.results['energy'] = _out['energy']
        self.results['charges'] = _out['charges']
        if do_forces:
            self.results['forces'] = _out['forces']


# =============================================================================
# Main Calculator Class
# =============================================================================
class ReferenceEnergyCalculator:
    """
    Calculate reference energies using machine learning potentials.
    
    Supported methods:
    - 'torchani' or 'ani2x': TorchANI with ANI-2x model
    - 'ani1x': TorchANI with ANI-1x model
    - 'ani1ccx': TorchANI with ANI-1ccx model
    - 'ani2xr': TorchANI with ANI-2xr model
    - 'ani2dr': TorchANI with ANI-2dr model
    - 'ani2xr_snn': TorchANI with ANI-2xr-Snn model
    - 'ani_mbis': TorchANI with ANI-mbis model
    - 'wb97m_d3' or 'aimnet2': AIMNet2 wB97M-D3 model
    - 'b97_3c': AIMNet2 B97-3c model
    - 'mace': MACE-OFF potential
    - 'xtb' or 'gfn2xtb': GFN2-xTB semi-empirical method
    
    Parameters
    ----------
    method : str
        Calculation method to use.
    work_dir : str
        Working directory for output files.
    device : str
        Device for computation ('cpu' or 'cuda'). Default 'cpu'.
    """
    
    SUPPORTED_METHODS = [
        'torchani', 'ani2x', 'ani1x', 'ani1ccx',
        'ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis',
        'wb97m_d3', 'aimnet2', 'b97_3c',
        'mace', 'xtb', 'gfn2xtb'
    ]
    
    def __init__(self, method: str = 'torchani', work_dir: str = './work',
                 device: str = 'cpu'):
        self.method = method.lower().replace('-', '_').replace(' ', '_')
        self.work_dir = create_work_dir(work_dir)
        self.device = device
        self.model = None
        self.calculator = None
        
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )
        
        self._setup_calculator()
    
    def _setup_calculator(self):
        """Initialize the ML potential calculator."""
        if self.method in ['torchani', 'ani2x']:
            self._setup_torchani('ANI2x')
        elif self.method == 'ani1x':
            self._setup_torchani('ANI1x')
        elif self.method == 'ani1ccx':
            self._setup_torchani('ANI1ccx')
        elif self.method == 'ani2xr':
            self._setup_torchani('ANI2xr')
        elif self.method == 'ani2dr':
            self._setup_torchani('ANI2dr')
        elif self.method == 'ani2xr_snn':
            self._setup_torchani('SnnANI2xr')
        elif self.method == 'ani_mbis':
            self._setup_torchani('ANImbis')
        elif self.method in ['wb97m_d3', 'aimnet2']:
            self._setup_aimnet2('wb97m_d3')
        elif self.method == 'b97_3c':
            self._setup_aimnet2('b97_3c')
        elif self.method == 'mace':
            self._setup_mace()
        elif self.method in ['xtb', 'gfn2xtb']:
            self._setup_xtb()
    
    def _setup_torchani(self, model_name: str):
        """
        Set up TorchANI calculator.
        
        All models are from the torchani package (https://github.com/aiqm/torchani).
        """
        import torch
        import torchani
        self.torch = torch
        
        model_map = {
            'ANI2x': torchani.models.ANI2x,
            'ANI1x': torchani.models.ANI1x,
            'ANI1ccx': torchani.models.ANI1ccx,
            'ANI2xr': torchani.models.ANI2xr,
            'ANI2dr': torchani.models.ANI2dr,
            'SnnANI2xr': torchani.models.SnnANI2xr,
            'ANImbis': torchani.models.ANImbis,
        }
        
        model_class = model_map[model_name]
        self.model = model_class(periodic_table_index=True).to(self.device)
        # ASE calculator for geometry optimization
        self.calculator = model_class().ase()
        logger.info(f"TorchANI {model_name} calculator initialized on {self.device}")
    
    def _setup_aimnet2(self, model_variant: str):
        """
        Set up AIMNet2 calculator.
        
        Clones the repo from https://github.com/pablo-arantes/AIMNet2 if needed.
        
        Parameters
        ----------
        model_variant : str
            'wb97m_d3' for wB97M-D3 or 'b97_3c' for B97-3c.
        """
        import torch
        self.torch = torch
        
        # Determine model filename
        if model_variant == 'wb97m_d3':
            model_candidates = ['aimnet2_wb97m-d3_ens.jpt', 'aimnet2_wb97m_0.jpt']
        else:
            model_candidates = ['aimnet2_b973c_ens.jpt', 'aimnet2_b973c_0.jpt']
        
        # Search paths
        search_dirs = [
            'AIMNet2/models',
            os.path.join(self.work_dir, 'AIMNet2/models'),
            '/content/AIMNet2/models',
        ]
        
        model_file = None
        for d in search_dirs:
            for fname in model_candidates:
                path = os.path.join(d, fname)
                if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
                    model_file = path
                    break
            if model_file:
                break
        
        # If not found, clone the repo
        if model_file is None:
            import subprocess
            clone_dir = os.path.join(self.work_dir, 'AIMNet2')
            if not os.path.exists(clone_dir):
                logger.info("Cloning AIMNet2 repository...")
                subprocess.run(
                    "git clone https://github.com/pablo-arantes/AIMNet2.git " + clone_dir,
                    shell=True, capture_output=True
                )
            # Check again
            for fname in model_candidates:
                path = os.path.join(clone_dir, 'models', fname)
                if os.path.isfile(path):
                    model_file = path
                    break
        
        if model_file is None:
            raise RuntimeError(
                f"Could not find AIMNet2 {model_variant} model. Try:\n"
                "  git clone https://github.com/pablo-arantes/AIMNet2.git"
            )
        
        logger.info(f"Loading AIMNet2 model from {model_file}")
        aimnet2_model = torch.jit.load(model_file, map_location=self.device)
        self.aimnet2_calc = AIMNet2Calculator(aimnet2_model)
        logger.info(f"AIMNet2 ({model_variant}) calculator initialized on {self.device}")
    
    def _setup_mace(self):
        """Set up MACE-OFF calculator."""
        try:
            import torch
            try:
                import torchvision
            except (ImportError, AttributeError):
                pass
            
            from mace.calculators import mace_off
            self.mace_calc = mace_off(model="medium", device=self.device)
            logger.info("MACE-OFF calculator initialized")
        except ImportError:
            raise ImportError(
                "MACE not installed. Install with:\n"
                "  pip install mace-torch\n"
                "If you get a torchvision error, try:\n"
                "  pip install --upgrade torchvision\n"
                "  pip install mace-torch"
            )
        except AttributeError as e:
            if 'torchvision' in str(e):
                raise ImportError(
                    "Circular import with torchvision. Fix by running:\n"
                    "  pip install --upgrade torchvision\n"
                    "Then restart the runtime and try again."
                )
            raise
    
    def _setup_xtb(self):
        """Set up GFN2-xTB calculator."""
        try:
            from xtb.ase.calculator import XTB
            self.xtb_calc = XTB(method="GFN2-xTB")
            logger.info("GFN2-xTB calculator initialized")
        except ImportError:
            raise ImportError(
                "xtb-python not installed. Install: conda install -c conda-forge xtb-python"
            )
    
    def _apply_dihedral_constraint(self, atoms, dihedral_indices: List[int]):
        """
        Apply a dihedral constraint to keep the target dihedral fixed during optimization.
        
        Uses ASE's FixInternals to constrain the dihedral angle to its current value.
        This is essential for dihedral scans: the dihedral must be kept fixed while
        all other degrees of freedom are relaxed.
        
        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms object.
        dihedral_indices : List[int]
            Four atom indices defining the dihedral to constrain.
        """
        from ase.constraints import FixInternals
        
        # Get the current dihedral angle value
        dihedral_indices_int = [int(a) for a in dihedral_indices]
        current_dihedral = atoms.get_dihedral(*dihedral_indices_int)
        
        # Apply the constraint: fix dihedral at its current value
        # Format: [[angle_value, [i, j, k, l]]]
        dihedral_constraint = [current_dihedral, dihedral_indices_int]
        constraint = FixInternals(dihedrals_deg=[dihedral_constraint])
        atoms.set_constraint(constraint)
        
        return current_dihedral
    
    def calculate_energy(self, mol_file: str, optimize: bool = True,
                        fmax: float = 0.05, steps: int = 200,
                        dihedral_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculate energy of a single conformer.
        
        Parameters
        ----------
        mol_file : str
            Path to MOL/PDB/XYZ file.
        optimize : bool
            Whether to optimize geometry. Default True.
        fmax : float
            Force convergence criterion for optimization.
        steps : int
            Maximum optimization steps.
        dihedral_indices : Optional[List[int]]
            Four atom indices defining the dihedral to constrain during optimization.
            If provided, the dihedral angle is kept fixed at its current value.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'energy' (Hartree or eV), 'energy_kcal' (kcal/mol), 'rho'.
        """
        from ase.io import read as ase_read
        atoms = ase_read(mol_file)
        
        if self.method in ['torchani', 'ani2x', 'ani1x', 'ani1ccx',
                           'ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis']:
            return self._calc_torchani(atoms, optimize, fmax, steps, dihedral_indices)
        elif self.method in ['wb97m_d3', 'aimnet2', 'b97_3c']:
            return self._calc_aimnet2(atoms, optimize, fmax, steps, dihedral_indices)
        elif self.method == 'mace':
            return self._calc_mace(atoms, optimize, fmax, steps, dihedral_indices)
        elif self.method in ['xtb', 'gfn2xtb']:
            return self._calc_xtb(atoms, optimize, fmax, steps, dihedral_indices)
    
    def _calc_torchani(self, atoms, optimize, fmax, steps, dihedral_indices=None) -> Dict[str, Any]:
        """
        Calculate energy using TorchANI.
        
        All ANI models (ANI-2x, ANI-1x, ANI-1ccx, ANI-2xr, ANI-2dr, 
        ANI-2xr-Snn, ANI-mbis) use the same torchani package.
        """
        import torch
        
        if optimize:
            from ase.optimize import LBFGS
            atoms.calc = self.calculator
            
            # Apply dihedral constraint before optimization
            if dihedral_indices is not None:
                self._apply_dihedral_constraint(atoms, dihedral_indices)
            
            opt = LBFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        
        # Calculate energy with the model directly (for ensemble disagreement)
        species = torch.tensor(
            [atoms.get_atomic_numbers()], dtype=torch.long, device=self.device
        )
        coordinates = torch.tensor(
            [atoms.get_positions()], dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        energy = self.model((species, coordinates)).energies
        energy_hartree = energy.item()
        
        # Calculate ensemble disagreement (rho) for reliability estimate
        rho = 0.0
        try:
            energies_members = []
            for member in self.model:
                e = member((species, coordinates)).energies.item()
                energies_members.append(e)
            if len(energies_members) > 1:
                rho = float(np.std(energies_members)) * HARTREE_TO_KCAL
        except Exception:
            pass
        
        return {
            'energy': energy_hartree,
            'energy_kcal': energy_hartree * HARTREE_TO_KCAL,
            'rho': rho,
            'positions': atoms.get_positions(),
        }
    
    def _calc_aimnet2(self, atoms, optimize, fmax, steps, dihedral_indices=None) -> Dict[str, Any]:
        """
        Calculate energy using AIMNet2.
        
        Uses the custom AIMNet2Calculator (proper ASE Calculator subclass)
        with torch.jit models from https://github.com/pablo-arantes/AIMNet2.
        """
        import torch
        from ase.optimize import LBFGS
        
        # Set up the calculator
        self.aimnet2_calc.do_reset()
        self.aimnet2_calc.set_charge(0)
        atoms.calc = self.aimnet2_calc
        
        if optimize:
            # Apply dihedral constraint before optimization
            if dihedral_indices is not None:
                self._apply_dihedral_constraint(atoms, dihedral_indices)
            
            with torch.jit.optimized_execution(False):
                opt = LBFGS(atoms, logfile=None)
                opt.run(fmax=fmax, steps=steps)
        
        # Final energy
        energy_hartree = atoms.get_potential_energy()
        
        return {
            'energy': energy_hartree,
            'energy_kcal': energy_hartree * HARTREE_TO_KCAL,
            'rho': 0.0,
            'positions': atoms.get_positions(),
        }
    
    def _calc_mace(self, atoms, optimize, fmax, steps, dihedral_indices=None) -> Dict[str, Any]:
        """Calculate energy using MACE-OFF."""
        from ase.optimize import LBFGS
        
        atoms.calc = self.mace_calc
        if optimize:
            # Apply dihedral constraint before optimization
            if dihedral_indices is not None:
                self._apply_dihedral_constraint(atoms, dihedral_indices)
            
            opt = LBFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        
        energy_ev = atoms.get_potential_energy()
        energy_kcal = energy_ev * 23.0609  # eV to kcal/mol
        return {
            'energy': energy_ev,
            'energy_kcal': energy_kcal,
            'rho': 0.0,
            'positions': atoms.get_positions(),
        }
    
    def _calc_xtb(self, atoms, optimize, fmax, steps, dihedral_indices=None) -> Dict[str, Any]:
        """Calculate energy using GFN2-xTB."""
        from xtb.ase.calculator import XTB
        from ase.optimize import LBFGS
        
        atoms.calc = XTB(method="GFN2-xTB")
        if optimize:
            # Apply dihedral constraint before optimization
            if dihedral_indices is not None:
                self._apply_dihedral_constraint(atoms, dihedral_indices)
            
            opt = LBFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        
        energy_ev = atoms.get_potential_energy()
        energy_kcal = energy_ev * 23.0609
        return {
            'energy': energy_ev,
            'energy_kcal': energy_kcal,
            'rho': 0.0,
            'positions': atoms.get_positions(),
        }
    
    def scan_dihedral(
        self,
        conformer_files: List[str],
        angles: Optional[List[float]] = None,
        optimize: bool = True,
        fmax: float = 0.05,
        steps: int = 200,
        dihedral_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate energies for a dihedral scan.
        
        Parameters
        ----------
        conformer_files : List[str]
            List of MOL/PDB files for each dihedral angle.
        angles : Optional[List[float]]
            Corresponding dihedral angles. If None, extracted from filenames.
        optimize : bool
            Whether to optimize each conformer (with dihedral constrained).
        dihedral_indices : Optional[List[int]]
            Four atom indices defining the dihedral. Required for constrained
            optimization. The dihedral angle is kept fixed at its value in each
            conformer file while all other degrees of freedom are relaxed.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'angles', 'energies_relative', 'rho_values', 'output_file'.
        """
        energies = []
        rho_values = []
        
        if angles is None:
            angles = []
            for f in conformer_files:
                basename = os.path.splitext(os.path.basename(f))[0]
                try:
                    angles.append(float(basename.split('_')[0]))
                except (ValueError, IndexError):
                    angles.append(float(basename))
        
        logger.info(f"Scanning {len(conformer_files)} conformers with {self.method}...")
        if dihedral_indices is not None:
            logger.info(f"  Dihedral constraint on atoms: {dihedral_indices}")
        
        for i, mol_file in enumerate(conformer_files):
            try:
                result = self.calculate_energy(
                    mol_file, optimize=optimize, fmax=fmax, steps=steps,
                    dihedral_indices=dihedral_indices
                )
                energies.append(result['energy_kcal'])
                rho_values.append(result.get('rho', 0.0))
                if (i + 1) % 5 == 0:
                    logger.info(f"  Processed {i+1}/{len(conformer_files)} conformers")
            except Exception as e:
                logger.warning(f"Error processing {mol_file}: {e}")
                energies.append(np.nan)
                rho_values.append(np.nan)
        
        energies_arr = np.array(energies)
        energies_relative = relative_energies(energies_arr)
        
        max_rho = max(rho_values) if rho_values else 0.0
        if max_rho > 0.6:
            logger.warning(f"High RHO={max_rho:.3f} > 0.6 kcal/mol. Results may be unreliable.")
        
        output_file = os.path.join(self.work_dir, f'{self.method}_scan.dat')
        write_energy_file(output_file, angles, energies_relative.tolist())
        
        logger.info(f"Dihedral scan complete. Energy range: {energies_relative.max():.3f} kcal/mol")
        
        return {
            'angles': angles,
            'energies': energies,
            'energies_relative': energies_relative.tolist(),
            'rho_values': rho_values,
            'output_file': output_file,
            'method': self.method,
        }
