# Getting Started with GPU Energy Modeling

This guide will help you set up and run the GPU energy modeling framework on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/gpu-energy-modeling.git
cd gpu-energy-modeling
```

2. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

This will install all necessary packages including NumPy, Pandas, Matplotlib, scikit-learn, and other dependencies.

## Running the Framework

### Running Benchmarks

To execute the benchmarks and collect simulated power data:

```bash
python scripts/run_apple_gpu_benchmarks.py
```

This script will:
- Run various GPU benchmarks (compute, memory, TBDR-specific)
- Collect simulated power and performance counter data
- Generate visualizations of power consumption
- Save results to the `data/benchmark_results` directory

You can specify particular benchmarks to run:

```bash
python scripts/run_apple_gpu_benchmarks.py --benchmark tile  # Run only tile memory benchmarks
python scripts/run_apple_gpu_benchmarks.py --benchmark visibility  # Run visibility determination benchmarks
python scripts/run_apple_gpu_benchmarks.py --benchmark unified  # Run unified memory benchmarks
python scripts/run_apple_gpu_benchmarks.py --benchmark compute  # Run compute benchmarks
python scripts/run_apple_gpu_benchmarks.py --benchmark memory  # Run memory benchmarks
```

### Validating the Energy Model

To validate the energy model against theoretical expectations and literature values:

```bash
python scripts/validate_model.py
```

This will:
- Test the linear energy model against known power scaling behaviors
- Validate TBDR access pattern energy implications
- Compare model results with published literature values
- Generate validation plots in the `data/validation` directory

## Exploring the Results

### Jupyter Notebooks

For interactive exploration of the data and models, you can use the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/1_GPU_Energy_Modeling_Demo.ipynb
```

This notebook demonstrates:
- Loading and analyzing benchmark results
- Training and evaluating energy models
- Visualizing power consumption patterns
- Performing what-if analysis

### Tableau Visualizations

The project includes support for creating interactive dashboards with Tableau:

```bash
# Generate data for Tableau
python scripts/generate_tableau_data.py
```

This will create CSV files in the `tableau/` directory that can be imported into Tableau Desktop or Tableau Public to create visualizations and dashboards.

For detailed instructions on creating Tableau visualizations with this project, see the [TABLEAU_GUIDE.md](TABLEAU_GUIDE.md) file.

## Project Structure

- `src/` - Core implementation of the energy modeling framework
  - `analysis/` - Energy efficiency metrics and visualization tools
  - `benchmarks/` - GPU benchmarks for different architectural features
  - `data_collection/` - Simulated data collection tools
  - `modeling/` - Energy model implementation
- `scripts/` - Utility scripts for running benchmarks and validation
- `notebooks/` - Jupyter notebooks for interactive analysis
- `data/` - Directory for benchmark results and validation output
- `resources/` - Technical documentation on GPU architecture and power

## Extending the Framework

The framework is designed to be extensible. Here are some ways you can build upon it:

1. **Adding new benchmarks:**
   - Extend the appropriate benchmark class in `src/benchmarks/`
   - Register your benchmark in `scripts/run_apple_gpu_benchmarks.py`

2. **Implementing new energy models:**
   - Create a new class that extends `BaseEnergyModel` in `src/modeling/energy_model.py`
   - Implement the required `train()` and `predict()` methods

3. **Creating custom visualizations:**
   - Add new visualization functions to `src/analysis/visualization.py`

## Troubleshooting

Common issues and their solutions:

1. **Missing dependencies:**
   - Make sure you've installed all required packages with `pip install -r requirements.txt`
   - For some environments, you may need to install additional system packages

2. **Data directory issues:**
   - The framework expects certain directories to exist. If you encounter errors about missing directories, create them manually:
     ```bash
     mkdir -p data/benchmark_results data/validation
     ```

3. **Display issues in headless environments:**
   - If running on a server without display, you may need to configure Matplotlib:
     ```python
     import matplotlib
     matplotlib.use('Agg')  # Use non-interactive backend
     ```

## Getting Help

If you encounter any issues or have questions, please:
1. Check the documentation in the `resources/` directory
2. Open an issue on the GitHub repository
3. Review existing issues for similar problems and solutions