# GPU Energy Modeling Framework Architecture

This document provides an overview of the GPU energy modeling framework architecture, explaining the key components and their interactions.

## System Architecture

The framework is designed around a modular architecture with the following main components:

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│   Benchmark     │     │  Data Collection  │     │  Energy Modeling   │
│   Framework     │────▶│     System        │────▶│     Engine         │
└─────────────────┘     └───────────────────┘     └────────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  Optimization   │◀────│    Analysis       │◀────│   Visualization    │
│  Suggestions    │     │     Tools         │     │      System        │
└─────────────────┘     └───────────────────┘     └────────────────────┘
```

### 1. Benchmark Framework

Located in: `src/benchmarks/`

The benchmark framework provides a structured approach to testing different aspects of GPU behavior:

- `base.py`: Defines the abstract `GPUBenchmark` class that all benchmarks extend
- `compute_benchmarks.py`: Implements compute-intensive benchmarks (matrix multiplication, convolution)
- `memory_benchmarks.py`: Implements memory access pattern benchmarks 
- `tbdr_benchmarks.py`: Implements tile-based deferred rendering specific benchmarks

Benchmarks are parameterized to allow testing different configurations and workloads.

### 2. Data Collection System

Located in: `src/data_collection/`

The data collection system is responsible for gathering metrics during benchmark execution:

- `collectors.py`: Implements collection of performance counters and power metrics
  - `SimulatedPowerCollector`: Provides simulated power data for development
  - `PerformanceCounterCollector`: Collects GPU performance counter data

The collectors save data in standardized formats (CSV, JSON) for later analysis.

### 3. Energy Modeling Engine

Located in: `src/modeling/`

The energy modeling engine builds predictive models from collected data:

- `energy_model.py`: Implements `LinearEnergyModel` for predicting power consumption
  - Trains on performance counter data
  - Provides feature importance analysis
  - Supports what-if analysis through prediction

### 4. Analysis Tools

Located in: `src/analysis/`

The analysis tools interpret the collected data and modeling results:

- `efficiency.py`: Calculates energy efficiency metrics
  - `calculate_energy_consumption`
  - `analyze_energy_efficiency`
  - `identify_efficiency_bottlenecks`
  - `what_if_analysis`

- `visualization.py`: Creates visualizations of energy/performance data
  - `plot_power_over_time`
  - `plot_component_breakdown`
  - `plot_efficiency_comparison`
  - `plot_model_feature_importance`

### 5. End-to-End Workflow

The primary workflow through the system is:

1. Run benchmarks with specific parameters
2. Collect performance counters and power data during execution
3. Build energy models from collected data
4. Analyze efficiency and identify optimization opportunities
5. Visualize results and generate recommendations

## Key Capabilities

### Micro-architectural Energy Modeling

The framework can build models that predict energy consumption based on architectural events (e.g., cache misses, memory bandwidth, compute utilization). This enables:

- Early power estimation without physical measurements
- Understanding component-level power breakdown
- Identification of power-critical subsystems

### Performance-per-Watt Analysis

By combining performance metrics with energy measurements, the framework calculates:

- Operations per watt
- Operations per joule
- Energy-delay product
- Energy proportionality

### Power Reduction Opportunity Identification

The analysis tools can identify:

- Inefficient memory access patterns
- Suboptimal compute patterns
- Opportunities for power-gating or clock-gating
- Software algorithm improvements for energy efficiency

## Running the Framework

The framework can be executed through:

1. **Scripted execution**:
   ```bash
   python scripts/run_apple_gpu_benchmarks.py
   ```

2. **Jupyter notebooks**:
   - `1_GPU_Energy_Modeling_Demo.ipynb`: Demonstrates the full workflow
   - `2_Apple_GPU_Architecture_Study.ipynb`: Analyzes Apple-specific architecture

The demonstration notebooks provide an interactive way to explore the framework capabilities and visualize results.