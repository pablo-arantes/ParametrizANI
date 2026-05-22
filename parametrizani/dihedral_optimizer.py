"""
ParametrizANI - Dihedral Optimizer
====================================

Optimize dihedral parameters via least-squares fitting.
Implements the Rotational Profiler algorithm (Rusu et al.) for computing
classical torsional parameters using linear least-squares regression.

Compatible with GROMOS, AMBER, OPLS, and CHARMM force field formats.
"""

import os
import logging
import math
import numpy as np
from typing import Dict, Any, Optional, List

from .utils import create_work_dir, write_energy_file, SMALL_REAL

logger = logging.getLogger(__name__)


class DihedralOptimizer:
    """
    Optimize dihedral parameters by fitting to a reference energy profile.

    Uses linear least-squares regression to determine optimal Fourier series
    coefficients for the dihedral potential:

        V(phi) = sum_n [ V_n * (1 + cos(n*phi - delta_n)) ]

    Parameters
    ----------
    max_terms : int
        Maximum number of Fourier terms (1-6). Default 6.
    work_dir : str
        Working directory for output files.
    allow_asymmetric_phase : bool
        If True, phase shifts are not constrained to 0/180. Default True.
    """

    def __init__(self, max_terms: int = 6, work_dir: str = './work',
                 allow_asymmetric_phase: bool = True):
        self.max_terms = min(max_terms, 6)
        self.work_dir = create_work_dir(work_dir)
        self.allow_asymmetric_phase = allow_asymmetric_phase
        self._setup_default_parameters()

    def _setup_default_parameters(self):
        """Set up default fitting parameters."""
        self.multiplicities = [0, 1, 1, 2, 2, 3, 6]
        self.phase_shifts = [1, 1, 1, 1, 1, 1, 1]
        self.phase_offsets = [0, 0, 120, 0, 120, 0, 0]
        self.is_active = [1, 1, 1, 1, 1, 1, 1]
        self.run_sequences = [
            [3], [0, 3], [0, 3, 5], [0, 3, 5, 6],
            [0, 1, 3, 5, 6], [0, 1, 2, 3, 5, 6],
        ]

    def _fit_dihedral(self, n_terms, ref_phis, ref_energies, mm_energies,
                      weights, multiplicities, phase_shifts, phase_offsets, is_active):
        """Core least-squares fitting routine."""
        n_points = len(ref_phis)
        n_active = sum(is_active)

        A = np.zeros((n_points, n_active))
        b = np.zeros(n_points)

        for i in range(n_points):
            phi_rad = math.radians(ref_phis[i])
            b[i] = (ref_energies[i] - mm_energies[i]) * weights[i]
            col = 0
            for j in range(n_terms):
                if is_active[j]:
                    n_mult = multiplicities[j]
                    offset_rad = math.radians(phase_offsets[j])
                    if phase_shifts[j] == 1:
                        A[i, col] = (1.0 + math.cos(n_mult * phi_rad - offset_rad)) * weights[i]
                    else:
                        A[i, col] = (1.0 - math.cos(n_mult * phi_rad - offset_rad)) * weights[i]
                    col += 1

        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_active)
            logger.warning("Least squares fitting failed")

        fitted = np.zeros(n_points)
        for i in range(n_points):
            phi_rad = math.radians(ref_phis[i])
            col = 0
            for j in range(n_terms):
                if is_active[j]:
                    n_mult = multiplicities[j]
                    offset_rad = math.radians(phase_offsets[j])
                    if phase_shifts[j] == 1:
                        fitted[i] += coeffs[col] * (1.0 + math.cos(n_mult * phi_rad - offset_rad))
                    else:
                        fitted[i] += coeffs[col] * (1.0 - math.cos(n_mult * phi_rad - offset_rad))
                    col += 1
            fitted[i] += mm_energies[i]

        return fitted, coeffs

    def run_optimization(
        self,
        ref_angles: List[float],
        ref_energies: List[float],
        mm_energies: Optional[List[float]] = None,
        atom_types: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full dihedral optimization.

        Parameters
        ----------
        ref_angles : List[float]
            Reference dihedral angles (degrees).
        ref_energies : List[float]
            Reference energies (kcal/mol).
        mm_energies : Optional[List[float]]
            MM energies (kcal/mol). If None, assumes zeros.
        atom_types : Optional[List[str]]
            Atom types for the 4 dihedral atoms.
        weights : Optional[List[float]]
            Fitting weights.

        Returns
        -------
        Dict[str, Any]
            Optimization results with 'best_fit', 'rmse', 'parameters', etc.
        """
        n_points = len(ref_angles)
        if mm_energies is None:
            mm_energies = [0.0] * n_points
        if weights is None:
            weights = [1.0] * n_points

        ref_min = min(ref_energies)
        mm_min = min(mm_energies)
        ref_norm = [e - ref_min for e in ref_energies]
        mm_norm = [e - mm_min for e in mm_energies] if abs(mm_min) > SMALL_REAL else mm_energies

        all_fitted = []
        all_coeffs = []
        all_rmse = []
        all_parameters = {}
        fitted_coefficients_list = []
        phase_offset_list = []
        multiplicity_list = []

        for run_idx, sequence in enumerate(self.run_sequences[:self.max_terms]):
            n_terms = len(sequence)
            sel_mult = [self.multiplicities[i] for i in sequence]
            sel_pshift = [self.phase_shifts[i] for i in sequence]
            sel_phase = [self.phase_offsets[i] for i in sequence]
            sel_active = [1] * n_terms

            fitted, coeffs = self._fit_dihedral(
                n_terms, ref_angles, ref_norm, mm_norm,
                weights, sel_mult, sel_pshift, sel_phase, sel_active
            )

            rmse = np.sqrt(np.mean((np.array(fitted) - np.array(ref_norm)) ** 2))
            all_fitted.append(fitted.tolist())
            all_coeffs.append(coeffs.tolist())
            all_rmse.append(rmse)

            params = {}
            for j, (coeff, mult, phase) in enumerate(zip(coeffs, sel_mult, sel_phase)):
                params[j + 1] = {
                    'V_n': float(coeff),
                    'phase': float(phase),
                    'periodicity': int(mult),
                }
                fitted_coefficients_list.append(float(coeff))
                phase_offset_list.append(float(phase))
                multiplicity_list.append(int(mult))
            all_parameters[run_idx + 1] = params
            logger.info(f"  {run_idx + 1} terms: RMSE = {rmse:.4f} kcal/mol")

        best_idx = self.max_terms - 1
        best_fit = all_fitted[best_idx]
        best_rmse = all_rmse[best_idx]
        best_params = all_parameters[best_idx + 1]

        output_file = os.path.join(self.work_dir, 'optimized_dihedral.dat')
        write_energy_file(output_file, ref_angles, best_fit)

        frcmod_params_by_terms = {
            n_terms: self._format_frcmod_params(params, atom_types)
            for n_terms, params in all_parameters.items()
        }
        frcmod_params = frcmod_params_by_terms.get(
            self.max_terms,
            self._format_frcmod_params(best_params, atom_types),
        )

        return {
            'angles': ref_angles,
            'energies_reference': ref_norm,
            'energies_mm': mm_norm,
            'energies_fitted': all_fitted,
            'best_fit': best_fit,
            'coefficients': all_coeffs,
            'parameters': all_parameters,
            'best_parameters': best_params,
            'rmse': best_rmse,
            'all_rmse': all_rmse,
            'atom_types': atom_types,
            'frcmod_params': frcmod_params,
            'frcmod_params_by_terms': frcmod_params_by_terms,
            'output_file': output_file,
            'fitted_coefficients_list': fitted_coefficients_list,
            'phase_offset_list': phase_offset_list,
            'multiplicity_list': multiplicity_list,
        }

    def get_parameters_for_terms(self, result: Dict[str, Any],
                                 n_terms: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Return the fitted parameter set for the requested number of Fourier terms."""
        if n_terms is None:
            return result['best_parameters']

        n_terms = int(n_terms)
        available = result.get('parameters', {})
        if n_terms not in available:
            raise ValueError(
                f"Requested {n_terms} terms, but available fits are: {sorted(available)}"
            )
        return available[n_terms]

    def format_frcmod_params(self, result: Dict[str, Any], n_terms: Optional[int] = None,
                             atom_types: Optional[List[str]] = None, idivf: int = 1) -> str:
        """Format the requested fitted parameter set in AMBER FRCMOD format."""
        params = self.get_parameters_for_terms(result, n_terms=n_terms)
        if atom_types is None:
            atom_types = result.get('atom_types')
        return self._format_frcmod_params(params, atom_types=atom_types, idivf=idivf)

    def _format_frcmod_params(self, params, atom_types=None, idivf=1):
        """Format parameters in AMBER FRCMOD format."""
        if atom_types is None:
            atom_types = ['X', 'X', 'X', 'X']
        type_str = f"{atom_types[0]}-{atom_types[1]}-{atom_types[2]}-{atom_types[3]}"
        lines = ["DIHE"]

        valid_terms = []
        for p in params.values():
            periodicity = int(p['periodicity'])
            if periodicity <= 0:
                continue
            valid_terms.append(p)

        for idx, p in enumerate(valid_terms):
            v_n = p['V_n']
            phase = p['phase']
            periodicity = abs(int(p['periodicity']))
            pn_sign = -periodicity if idx < len(valid_terms) - 1 else periodicity
            line = f"{type_str}   {idivf}   {v_n:10.4f}  {phase:8.2f}  {pn_sign:6.1f}"
            lines.append(line)
        return "\n".join(lines)

    def write_frcmod(self, result, output_file=None, idivf=1, n_terms=None):
        """Write optimized parameters to FRCMOD file."""
        if output_file is None:
            output_file = os.path.join(self.work_dir, 'optimized.frcmod')
        frcmod_content = self.format_frcmod_params(
            result, n_terms=n_terms, atom_types=result.get('atom_types'), idivf=idivf
        )
        with open(output_file, 'w') as f:
            f.write("Remark: ParametrizANI optimized dihedral parameters\n")
            f.write(frcmod_content)
            f.write("\n\n")
        logger.info(f"FRCMOD file written: {output_file}")
        return output_file

    def calculate_dihedral_energy(self, angles, parameters, mm_energies=None):
        """Calculate dihedral energy for given angles using fitted parameters."""
        if mm_energies is None:
            mm_energies = [0.0] * len(angles)
        values = []
        for i, angle in enumerate(angles):
            e_dihedral = 0.0
            for term in parameters.values():
                v_n = term['V_n']
                phase = term['phase']
                n = term['periodicity']
                e_dihedral += v_n * (1.0 + math.cos(math.radians(n * angle - phase)))
            values.append(e_dihedral + mm_energies[i])
        return values
