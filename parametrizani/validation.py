"""
ParametrizANI - Parameter Validation
======================================

Validate optimized dihedral parameters against reference data.
Provides metrics, quality ratings, and visualization.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from .utils import create_work_dir

logger = logging.getLogger(__name__)


class ParameterValidator:
    """
    Validate optimized dihedral parameters.
    
    Computes RMSE, MAE, R\u00b2, correlation and generates reports/plots.
    
    Parameters
    ----------
    work_dir : str
        Working directory for output files.
    """
    
    QUALITY_THRESHOLDS = {
        'Excellent': 0.25,
        'Good': 0.50,
        'Acceptable': 1.00,
        'Poor': float('inf'),
    }
    
    def __init__(self, work_dir: str = './work'):
        self.work_dir = create_work_dir(work_dir)
    
    def validate_parameters(
        self, angles: List[float], ref_energies: List[float],
        fitted_energies: List[float], mm_energies: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate fitted parameters against reference.
        
        Returns
        -------
        Dict[str, Any]
            'rmse', 'mae', 'r_squared', 'correlation', 'max_error', 'quality',
            'report_file', 'plot_file'.
        """
        ref = np.array(ref_energies)
        fitted = np.array(fitted_energies)
        
        rmse = np.sqrt(np.mean((ref - fitted) ** 2))
        mae = np.mean(np.abs(ref - fitted))
        max_error = np.max(np.abs(ref - fitted))
        
        ss_res = np.sum((ref - fitted) ** 2)
        ss_tot = np.sum((ref - np.mean(ref)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        correlation = np.corrcoef(ref, fitted)[0, 1] if len(ref) > 1 else 0.0
        
        quality = 'Poor'
        for rating, threshold in self.QUALITY_THRESHOLDS.items():
            if rmse <= threshold:
                quality = rating
                break
        
        logger.info(f"Validation: RMSE={rmse:.4f}, R\u00b2={r_squared:.4f}, Quality={quality}")
        
        report = self._generate_report(
            angles, ref, fitted, mm_energies, rmse, mae, r_squared, correlation, max_error, quality
        )
        report_file = os.path.join(self.work_dir, 'validation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        plot_file = self._generate_plot(angles, ref, fitted, mm_energies, labels, quality, rmse)
        
        return {
            'rmse': rmse, 'mae': mae, 'r_squared': r_squared,
            'correlation': correlation, 'max_error': max_error,
            'quality': quality, 'report_file': report_file, 'plot_file': plot_file,
        }
    
    def compare_methods(self, angles, energy_profiles, reference_key=None):
        """Compare multiple energy profiles."""
        results = {}
        if reference_key is None:
            reference_key = list(energy_profiles.keys())[0]
        ref = np.array(energy_profiles[reference_key])
        for name, energies in energy_profiles.items():
            if name == reference_key:
                continue
            pred = np.array(energies)
            rmse = np.sqrt(np.mean((ref - pred) ** 2))
            mae = np.mean(np.abs(ref - pred))
            ss_res = np.sum((ref - pred) ** 2)
            ss_tot = np.sum((ref - np.mean(ref)) ** 2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            results[name] = {'rmse': rmse, 'mae': mae, 'r_squared': r_squared}
        plot_file = self._generate_comparison_plot(angles, energy_profiles, reference_key)
        return {'reference': reference_key, 'metrics': results, 'plot_file': plot_file}
    
    def _generate_report(self, angles, ref, fitted, mm_energies, rmse, mae, r_squared, correlation, max_error, quality):
        lines = [
            "=" * 60, "ParametrizANI - Validation Report", "=" * 60, "",
            f"Quality Rating: {quality}", "", "Metrics:",
            f"  RMSE:           {rmse:.4f} kcal/mol",
            f"  MAE:            {mae:.4f} kcal/mol",
            f"  Max Error:      {max_error:.4f} kcal/mol",
            f"  R\u00b2:             {r_squared:.4f}",
            f"  Correlation:    {correlation:.4f}", "",
            "Per-point comparison:",
            f"  {'Angle':>8s} {'Reference':>10s} {'Fitted':>10s} {'Error':>10s}",
        ]
        for angle, r, f in zip(angles, ref, fitted):
            lines.append(f"  {angle:8.1f} {r:10.4f} {f:10.4f} {abs(r-f):10.4f}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)
    
    def _generate_plot(self, angles, ref, fitted, mm_energies, labels, quality, rmse):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ref_label = labels.get('reference', 'Reference') if labels else 'Reference'
            fit_label = labels.get('fitted', 'Optimized') if labels else 'Optimized'
            ax.plot(angles, ref, 'o-', linewidth=1.5, markersize=4, label=ref_label, color='#2196F3')
            ax.plot(angles, fitted, 's-', linewidth=1.5, markersize=4, label=fit_label, color='#F44336')
            if mm_energies is not None:
                mm_label = labels.get('mm', 'MM (original)') if labels else 'MM (original)'
                ax.plot(angles, mm_energies, '^--', linewidth=1.0, markersize=3, label=mm_label, color='#4CAF50', alpha=0.7)
            ax.set_xlabel('Dihedral Angle (degrees)', fontsize=12)
            ax.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
            ax.set_title(f'Dihedral Parametrization - {quality} (RMSE: {rmse:.4f} kcal/mol)', fontsize=13)
            ax.legend(fontsize=11, frameon=True)
            ax.set_xticks(np.arange(min(angles), max(angles) + 1, 60))
            plot_file = os.path.join(self.work_dir, 'validation_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_file
        except ImportError:
            return ""
    
    def _generate_comparison_plot(self, angles, energy_profiles, reference_key):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
            for i, (name, energies) in enumerate(energy_profiles.items()):
                lw = 2.0 if name == reference_key else 1.5
                ax.plot(angles, energies, 'o-', linewidth=lw, markersize=4, label=name, color=colors[i % len(colors)])
            ax.set_xlabel('Dihedral Angle (degrees)', fontsize=12)
            ax.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
            ax.set_title('Energy Profile Comparison', fontsize=13)
            ax.legend(fontsize=11, frameon=True)
            plot_file = os.path.join(self.work_dir, 'comparison_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_file
        except ImportError:
            return ""
