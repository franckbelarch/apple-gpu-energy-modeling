# Validation Data

This directory contains validation results from the GPU energy model verification process.

When you run `python scripts/validate_model.py` or the validation notebook, the following files will be generated here:

- `compute_scaling_validation.png` - Shows how power scales with compute utilization
- `memory_scaling_validation.png` - Shows how power scales with memory bandwidth
- `tbdr_access_pattern_validation.png` - Shows energy differences between tile memory access patterns
- `literature_comparison.png` - Compares model results with published literature values

These validation plots help verify that the energy model accurately captures the expected relationships between GPU activity and power consumption.

See the `VERIFICATION.md` document in the repository root for a detailed explanation of the validation methodology.