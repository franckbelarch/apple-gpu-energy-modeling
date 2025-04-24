# GPU Energy Modeling Project Execution Order

This document outlines the proper sequence for running the notebooks and scripts in the GPU Energy Modeling project. Following this order ensures that all dependencies are properly initialized and that data flows correctly between components.

## 1. Setup and Installation

Before running any code, make sure you've completed the setup as described in `GETTING_STARTED.md`:

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p data/benchmark_results data/validation
```

## 2. Core Notebooks Sequence

The notebooks should be run in the following order:

### 2.1. Foundational Notebooks

1. **1_GPU_Energy_Modeling_Demo.ipynb**
   - This notebook sets up the core components of the energy modeling framework
   - Demonstrates basic matrix multiplication benchmark
   - Trains the initial energy model
   - If model training shows perfect results (RMSE = 0, R² = 1), check that:
     - Data variation exists in your input features
     - You're not overfitting with too many features for too few data points
     - The power data has realistic variation

2. **2_Apple_GPU_Architecture_Study.ipynb**
   - Introduces Apple's GPU architecture concepts
   - Explores TBDR (Tile-Based Deferred Rendering)
   - Examines unified memory advantages

### 2.2. Validation and Case Studies

3. **validation_demo.ipynb**
   - Validates the energy model against theoretical expectations
   - Compares with literature values
   - Must be run after the foundational notebooks, as it builds on the basic model

4. **memory_access_optimization.ipynb**
   - Analyzes different memory access patterns
   - **Important**: Run all cells in order from top to bottom
   - Each cell depends on variables defined in previous cells
   - Benchmarks are defined and run in cell 5
   - The graph in cell 11 requires data from earlier cells

5. **shader_efficiency_optimization.ipynb**
   - Examines shader optimization techniques
   - Compares different shader implementations
   - **Important**: Run all cells in order from top to bottom

## 3. Scripts Execution Order

The Python scripts should be run in this sequence:

1. **validate_model.py**
   ```bash
   python scripts/validate_model.py
   ```
   - Ensures the energy model is validated against theoretical expectations
   - Generates validation plots in data/validation directory

2. **run_apple_gpu_benchmarks.py**
   ```bash
   python scripts/run_apple_gpu_benchmarks.py
   ```
   - Runs all benchmarks and collects performance data
   - Generates benchmark results in data/benchmark_results directory
   - Can be run with specific benchmark flags (see GETTING_STARTED.md)

3. **generate_tableau_data.py**
   ```bash
   python scripts/generate_tableau_data.py
   ```
   - Creates Tableau-ready data files based on benchmark results
   - Should only be run after benchmarks have been executed

## 4. Troubleshooting Common Issues

### 4.1. Missing Data or Blank Graphs

If you encounter blank graphs or missing data:
- Ensure you've run all cells in the correct order
- Check for error messages in earlier cells
- Verify that benchmark data was successfully generated
- Look for memory_accessed values in the debug output

### 4.2. Perfect Model Results (RMSE = 0, R² = 1)

If your model shows perfect validation metrics:
- Check that your input data has sufficient variation
- Ensure you're not using the same data for training and testing
- Verify that the power data simulation is adding realistic variation
- Try reducing the number of features or increasing the training data size

### 4.3. Execution Time Issues

Some benchmarks might take time to run:
- Memory benchmarks with large buffer sizes can be slow
- Power collection with high sampling rates can be memory intensive
- If needed, reduce the benchmark parameters (buffer_size_mb, iterations) for faster execution

## 5. Extending the Framework

When adding new benchmarks or analysis notebooks:
1. Follow the same structure as existing benchmarks
2. Ensure all metrics are properly measured and returned
3. Include memory_accessed and execution_time in benchmark results
4. Add new notebooks to this execution order document

By following this execution order, you'll ensure that all components of the GPU Energy Modeling project work correctly together.