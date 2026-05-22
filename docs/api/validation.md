# ParameterValidator

Validate optimized parameters with quality metrics and visualization.

## Usage

```python
from parametrizani import ParameterValidator

validator = ParameterValidator('./work')
validation = validator.validate_parameters(
    ref_angles, ref_energies, fitted_energies
)

print(f"Quality: {validation['quality']}")
print(f"RMSE: {validation['rmse']:.4f} kcal/mol")
print(f"MAE: {validation['mae']:.4f} kcal/mol")
print(f"R²: {validation['r_squared']:.4f}")
```

## Quality Ratings

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | ≤ 0.25 | Near-QM accuracy |
| Good | ≤ 0.50 | Suitable for most applications |
| Acceptable | ≤ 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

## API Reference

::: parametrizani.validation.ParameterValidator
    options:
      members:
        - __init__
        - validate_parameters
      show_root_heading: true
      heading_level: 3
