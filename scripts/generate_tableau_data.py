#!/usr/bin/env python3
"""
Generate sample data for Tableau visualization from the GPU energy modeling framework.
This script runs benchmarks and formats the results in CSV files ready for Tableau.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.compute_benchmarks import MatrixMultiplication, ConvolutionBenchmark
from src.benchmarks.memory_benchmarks import MemoryCopy, RandomAccess
from src.benchmarks.tbdr_benchmarks import TileMemoryBenchmark, VisibilityDeterminationBenchmark, UnifiedMemoryBenchmark
from src.data_collection.collectors import SimulatedPowerCollector
from src.analysis.efficiency import calculate_energy_consumption, analyze_energy_efficiency
from src.modeling.energy_model import LinearEnergyModel


def generate_compute_benchmark_data():
    """Generate data from compute benchmarks for Tableau"""
    print("Generating compute benchmark data...")
    
    # Create benchmark instances
    matmul = MatrixMultiplication()
    conv = ConvolutionBenchmark()
    
    # Run matrix multiplication benchmarks with different sizes
    matrix_sizes = [512, 1024, 2048, 4096]
    matmul_results = []
    
    for size in matrix_sizes:
        # Run benchmark
        result = matmul.run({'matrix_size': size})
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create activity pattern
        activity_pattern = np.concatenate([
            np.linspace(0.2, 0.8, num_samples // 4),  # Ramp up
            np.ones(num_samples // 2) * 0.8,          # Steady state
            np.linspace(0.8, 0.2, num_samples // 4)   # Ramp down
        ])
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Add to results
        matmul_results.append({
            'benchmark': 'MatrixMultiplication',
            'parameter': f'size={size}',
            'matrix_size': size,
            'operations': result.get('operations', 0),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'operations_per_watt': result.get('operations', 0) / power_df['total_power'].mean(),
            'operations_per_joule': result.get('operations', 0) / energy if energy > 0 else 0,
        })
    
    # Run convolution benchmarks with different configurations
    filter_sizes = [3, 5, 7, 9]
    conv_results = []
    
    for filter_size in filter_sizes:
        # Run benchmark
        result = conv.run({
            'input_size': 224,
            'kernel_size': filter_size,
            'channels': 3,
            'filters': 64
        })
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create activity pattern - higher activity for larger filters
        max_activity = 0.6 + (filter_size / 20.0)  # Scale with filter size
        activity_pattern = np.concatenate([
            np.linspace(0.2, max_activity, num_samples // 4),
            np.ones(num_samples // 2) * max_activity,
            np.linspace(max_activity, 0.2, num_samples // 4)
        ])
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Add to results
        conv_results.append({
            'benchmark': 'Convolution',
            'parameter': f'filter={filter_size}',
            'filter_size': filter_size,
            'operations': result.get('operations', 0),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'operations_per_watt': result.get('operations', 0) / power_df['total_power'].mean(),
            'operations_per_joule': result.get('operations', 0) / energy if energy > 0 else 0,
        })
    
    # Create combined DataFrame
    compute_df = pd.DataFrame(matmul_results + conv_results)
    
    # Save to CSV for Tableau
    os.makedirs('./tableau', exist_ok=True)
    compute_df.to_csv('./tableau/compute_benchmarks.csv', index=False)
    
    print(f"Saved compute benchmark data to tableau/compute_benchmarks.csv")
    return compute_df


def generate_memory_benchmark_data():
    """Generate data from memory benchmarks for Tableau"""
    print("Generating memory benchmark data...")
    
    # Create benchmark instances
    memcopy = MemoryCopy()
    random_access = RandomAccess()
    
    # Run memory copy benchmarks with different buffer sizes
    buffer_sizes = [64, 128, 256, 512, 1024]
    memcopy_results = []
    
    for size in buffer_sizes:
        # Run benchmark
        result = memcopy.run({'buffer_size_mb': size, 'iterations': 5})
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create activity pattern - higher activity for larger buffers
        max_activity = 0.5 + (size / 2048.0)  # Scale with buffer size
        activity_pattern = np.concatenate([
            np.linspace(0.2, max_activity, num_samples // 4),
            np.ones(num_samples // 2) * max_activity,
            np.linspace(max_activity, 0.2, num_samples // 4)
        ])
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Calculate memory bandwidth
        memory_bandwidth = result.get('memory_copied', 0) / result.get('mean_execution_time', 1)
        
        # Add to results
        memcopy_results.append({
            'benchmark': 'MemoryCopy',
            'parameter': f'size={size}MB',
            'buffer_size_mb': size,
            'memory_copied_mb': result.get('memory_copied', 0) / (1024 * 1024),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'memory_bandwidth_mbs': memory_bandwidth / (1024 * 1024),
            'bandwidth_per_watt': memory_bandwidth / power_df['total_power'].mean(),
            'bandwidth_per_joule': memory_bandwidth / energy if energy > 0 else 0,
        })
    
    # Run random access benchmarks with different array sizes
    array_sizes = [32, 64, 128, 256, 512]
    random_access_results = []
    
    for size in array_sizes:
        # Run benchmark
        result = random_access.run({
            'array_size_mb': size, 
            'access_count': size * 20000  # Scale access count with array size
        })
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create activity pattern with random variations to simulate random access pattern
        base_activity = 0.5 + (size / 1024.0)  # Scale with array size
        activity_pattern = np.random.normal(base_activity, 0.1, num_samples)
        activity_pattern = np.clip(activity_pattern, 0.2, 0.9)
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Calculate memory bandwidth (approximation)
        memory_bandwidth = result.get('memory_accessed', 0) / result.get('mean_execution_time', 1)
        
        # Add to results
        random_access_results.append({
            'benchmark': 'RandomAccess',
            'parameter': f'size={size}MB',
            'array_size_mb': size,
            'memory_accessed_mb': result.get('memory_accessed', 0) / (1024 * 1024),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'memory_bandwidth_mbs': memory_bandwidth / (1024 * 1024),
            'bandwidth_per_watt': memory_bandwidth / power_df['total_power'].mean(),
            'bandwidth_per_joule': memory_bandwidth / energy if energy > 0 else 0,
        })
    
    # Create combined DataFrame
    memory_df = pd.DataFrame(memcopy_results + random_access_results)
    
    # Save to CSV for Tableau
    memory_df.to_csv('./tableau/memory_benchmarks.csv', index=False)
    
    print(f"Saved memory benchmark data to tableau/memory_benchmarks.csv")
    return memory_df


def generate_tbdr_benchmark_data():
    """Generate data from TBDR benchmarks for Tableau"""
    print("Generating TBDR benchmark data...")
    
    # Create benchmark instances
    tile_memory = TileMemoryBenchmark()
    visibility = VisibilityDeterminationBenchmark()
    unified_memory = UnifiedMemoryBenchmark()
    
    # Run tile memory benchmarks with different access patterns
    access_patterns = ['sequential', 'random', 'alternating']
    tile_memory_results = []
    
    for pattern in access_patterns:
        # Run benchmark
        result = tile_memory.run({
            'tile_size': 32,
            'tile_count': 100,
            'access_pattern': pattern,
            'overdraw': 1.0
        })
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create appropriate activity pattern based on access pattern
        if pattern == 'sequential':
            # Sequential is efficient - steady activity
            activity_factor = 0.7
            activity_pattern = np.ones(num_samples) * activity_factor
        elif pattern == 'random':
            # Random access is inefficient - higher power
            activity_factor = 0.9
            activity_pattern = np.random.normal(activity_factor, 0.1, num_samples)
            activity_pattern = np.clip(activity_pattern, 0.5, 1.0)
        else:  # alternating
            # Alternating is in between
            activity_factor = 0.8
            activity_pattern = np.sin(np.linspace(0, 10, num_samples)) * 0.1 + activity_factor
            activity_pattern = np.clip(activity_pattern, 0.5, 1.0)
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Add to results
        tile_memory_results.append({
            'benchmark': 'TileMemory',
            'parameter': pattern,
            'access_pattern': pattern,
            'tile_count': 100,
            'operations': result.get('operations', 0),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'operations_per_watt': result.get('operations', 0) / power_df['total_power'].mean(),
            'operations_per_joule': result.get('operations', 0) / energy if energy > 0 else 0,
        })
    
    # Run visibility determination benchmarks with different depth complexity
    depth_values = [2, 5, 10]
    visibility_results = []
    
    for depth in depth_values:
        # Run benchmark
        result = visibility.run({
            'tile_size': 32,
            'tile_count': 100,
            'depth_complexity': depth,
            'occlusion_rate': 0.7
        })
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create activity pattern for power simulation
        activity_factor = 0.5 + (depth / 20)  # Scale with depth complexity
        
        # Create pattern - initial spike for setup, then processing
        activity_pattern = np.concatenate([
            np.linspace(0.3, activity_factor, num_samples // 4),      # Ramp up 
            np.ones(num_samples // 4) * activity_factor,              # Process geometry
            np.ones(num_samples // 4) * (activity_factor * 0.7),      # HSR savings
            np.linspace(activity_factor * 0.7, 0.3, num_samples // 4) # Wind down
        ])
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Add to results
        visibility_results.append({
            'benchmark': 'VisibilityDetermination',
            'parameter': f'depth={depth}',
            'depth_complexity': depth,
            'occlusion_rate': 0.7,
            'total_fragments': result.get('total_fragments', 0),
            'visible_fragments': result.get('visible_fragments', 0),
            'occluded_fragments': result.get('occluded_fragments', 0),
            'visibility_efficiency': result.get('occluded_fragments', 0) / result.get('total_fragments', 1),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'estimated_energy_saved': result.get('visibility_efficiency', 0) * 100  # As percentage
        })
    
    # Run unified memory benchmarks with different sharing patterns
    sharing_patterns = ['alternating', 'producer_consumer', 'mixed_access']
    unified_memory_results = []
    
    for pattern in sharing_patterns:
        # Run benchmark
        result = unified_memory.run({
            'buffer_size_mb': 64,
            'sharing_pattern': pattern,
            'iterations': 10
        })
        
        # Collect simulated power data
        power_collector = SimulatedPowerCollector(output_dir='./data/tableau')
        duration = 5.0  # seconds
        num_samples = int(duration / power_collector.sampling_interval)
        
        # Create appropriate activity pattern based on sharing pattern
        if pattern == 'alternating':
            # Alternating CPU/GPU usage - sawtooth pattern
            x = np.linspace(0, 10, num_samples)
            activity_pattern = np.abs((x % 1.0) - 0.5) * 0.8 + 0.4
        elif pattern == 'producer_consumer':
            # Producer/consumer - step pattern (CPU then GPU)
            activity_pattern = np.ones(num_samples) * 0.6
            for i in range(5):
                start = int(i * num_samples / 5)
                mid = int((i + 0.5) * num_samples / 5)
                end = int((i + 1) * num_samples / 5)
                activity_pattern[start:mid] = 0.4  # CPU (lower power)
                activity_pattern[mid:end] = 0.8  # GPU (higher power)
        else:
            # Mixed access - more random pattern
            activity_pattern = np.random.normal(0.6, 0.1, num_samples)
            activity_pattern = np.clip(activity_pattern, 0.3, 0.9)
        
        # Collect power data
        power_data = power_collector.collect_for_duration(duration, activity_pattern)
        power_df = pd.DataFrame(power_data)
        
        # Calculate energy
        energy = calculate_energy_consumption(power_df)
        
        # Add to results
        unified_memory_results.append({
            'benchmark': 'UnifiedMemory',
            'parameter': pattern,
            'sharing_pattern': pattern,
            'buffer_size_mb': 64,
            'cpu_operations': result.get('cpu_operations', 0),
            'gpu_operations': result.get('gpu_operations', 0),
            'total_operations': result.get('total_operations', 0),
            'transfers_saved': result.get('transfers_saved', 0),
            'execution_time': result.get('mean_execution_time', 0),
            'energy_consumption': energy,
            'operations_per_joule': result.get('total_operations', 0) / energy if energy > 0 else 0,
            'estimated_energy_saved': result.get('estimated_energy_saved', 0) / energy if energy > 0 else 0,
        })
    
    # Create combined DataFrame
    tbdr_df = pd.DataFrame(tile_memory_results + visibility_results + unified_memory_results)
    
    # Save to CSV for Tableau
    tbdr_df.to_csv('./tableau/tbdr_benchmarks.csv', index=False)
    
    print(f"Saved TBDR benchmark data to tableau/tbdr_benchmarks.csv")
    return tbdr_df


def generate_component_power_data():
    """Generate component power breakdown data for Tableau"""
    print("Generating component power breakdown data...")
    
    # Create simulated workloads with different component utilization patterns
    workloads = [
        {
            'name': 'compute_intensive',
            'description': 'Compute-Intensive Workload',
            'compute_power_ratio': 0.65,
            'memory_power_ratio': 0.25,
            'io_power_ratio': 0.10
        },
        {
            'name': 'memory_intensive',
            'description': 'Memory-Intensive Workload',
            'compute_power_ratio': 0.35,
            'memory_power_ratio': 0.55,
            'io_power_ratio': 0.10
        },
        {
            'name': 'balanced',
            'description': 'Balanced Workload',
            'compute_power_ratio': 0.45,
            'memory_power_ratio': 0.40,
            'io_power_ratio': 0.15
        },
        {
            'name': 'idle',
            'description': 'Idle State',
            'compute_power_ratio': 0.20,
            'memory_power_ratio': 0.15,
            'io_power_ratio': 0.65
        }
    ]
    
    # Generate component power data for each workload
    power_data = []
    
    for workload in workloads:
        # Base power levels
        base_power = 10.0  # Watts
        
        if workload['name'] == 'compute_intensive':
            base_power = 25.0
        elif workload['name'] == 'memory_intensive':
            base_power = 22.0
        elif workload['name'] == 'balanced':
            base_power = 20.0
        elif workload['name'] == 'idle':
            base_power = 5.0
        
        # Generate multiple samples for this workload
        for i in range(10):  # 10 samples per workload
            # Add some random variation
            variation = np.random.normal(0, 0.05)  # 5% random variation
            adjusted_power = base_power * (1 + variation)
            
            # Calculate component power based on ratios
            compute_power = adjusted_power * workload['compute_power_ratio']
            memory_power = adjusted_power * workload['memory_power_ratio']
            io_power = adjusted_power * workload['io_power_ratio']
            
            # Add to data
            power_data.append({
                'workload': workload['name'],
                'workload_description': workload['description'],
                'sample_id': i,
                'total_power': adjusted_power,
                'compute_power': compute_power,
                'memory_power': memory_power,
                'io_power': io_power
            })
    
    # Create DataFrame
    component_df = pd.DataFrame(power_data)
    
    # Calculate percentage columns for Tableau
    component_df['compute_percent'] = component_df['compute_power'] / component_df['total_power'] * 100
    component_df['memory_percent'] = component_df['memory_power'] / component_df['total_power'] * 100
    component_df['io_percent'] = component_df['io_power'] / component_df['total_power'] * 100
    
    # Save to CSV for Tableau
    component_df.to_csv('./tableau/component_power.csv', index=False)
    
    print(f"Saved component power data to tableau/component_power.csv")
    return component_df


def generate_optimization_scenario_data():
    """Generate what-if optimization scenario data for Tableau"""
    print("Generating optimization scenario data...")
    
    # Define optimization scenarios
    scenarios = [
        {
            'name': 'baseline',
            'description': 'Baseline (No Optimization)',
            'compute_scale': 1.0,
            'memory_scale': 1.0,
            'io_scale': 1.0
        },
        {
            'name': 'compute_opt',
            'description': 'Compute Optimization',
            'compute_scale': 0.7,  # 30% reduction in compute power
            'memory_scale': 1.0,
            'io_scale': 1.0
        },
        {
            'name': 'memory_opt',
            'description': 'Memory Access Optimization',
            'compute_scale': 1.0,
            'memory_scale': 0.6,  # 40% reduction in memory power
            'io_scale': 1.0
        },
        {
            'name': 'full_opt',
            'description': 'Full System Optimization',
            'compute_scale': 0.8,  # 20% reduction in compute
            'memory_scale': 0.7,  # 30% reduction in memory
            'io_scale': 0.9  # 10% reduction in I/O
        }
    ]
    
    # Generate data for different workloads under different scenarios
    workloads = ['matrix_multiply', 'convolution', 'memory_intensive', 'mixed']
    scenario_data = []
    
    # Base power values for each workload
    base_powers = {
        'matrix_multiply': {'compute': 15.0, 'memory': 5.0, 'io': 2.0},
        'convolution': {'compute': 18.0, 'memory': 7.0, 'io': 2.5},
        'memory_intensive': {'compute': 8.0, 'memory': 12.0, 'io': 3.0},
        'mixed': {'compute': 12.0, 'memory': 10.0, 'io': 3.5}
    }
    
    # Performance impact of optimizations (some optimizations might reduce performance)
    perf_impact = {
        'baseline': {'matrix_multiply': 1.0, 'convolution': 1.0, 'memory_intensive': 1.0, 'mixed': 1.0},
        'compute_opt': {'matrix_multiply': 0.95, 'convolution': 0.97, 'memory_intensive': 1.02, 'mixed': 0.99},
        'memory_opt': {'matrix_multiply': 1.05, 'convolution': 1.03, 'memory_intensive': 0.92, 'mixed': 1.0},
        'full_opt': {'matrix_multiply': 0.98, 'convolution': 1.0, 'memory_intensive': 0.95, 'mixed': 0.97}
    }
    
    for workload in workloads:
        base_power = base_powers[workload]
        
        for scenario in scenarios:
            # Apply scenario scaling factors
            compute_power = base_power['compute'] * scenario['compute_scale']
            memory_power = base_power['memory'] * scenario['memory_scale']
            io_power = base_power['io'] * scenario['io_scale']
            total_power = compute_power + memory_power + io_power
            
            # Calculate performance impact
            performance = perf_impact[scenario['name']][workload]
            
            # Calculate energy efficiency (normalize to baseline)
            baseline_power = sum(base_powers[workload].values())
            power_reduction = baseline_power - total_power
            power_reduction_percent = (power_reduction / baseline_power) * 100
            
            # Energy efficiency metric (higher is better)
            energy_efficiency = performance / (total_power / baseline_power)
            
            # Add to data
            scenario_data.append({
                'workload': workload,
                'scenario': scenario['name'],
                'scenario_description': scenario['description'],
                'compute_power': compute_power,
                'memory_power': memory_power,
                'io_power': io_power,
                'total_power': total_power,
                'performance': performance,
                'power_reduction': power_reduction,
                'power_reduction_percent': power_reduction_percent,
                'energy_efficiency': energy_efficiency
            })
    
    # Create DataFrame
    scenario_df = pd.DataFrame(scenario_data)
    
    # Save to CSV for Tableau
    scenario_df.to_csv('./tableau/optimization_scenarios.csv', index=False)
    
    print(f"Saved optimization scenario data to tableau/optimization_scenarios.csv")
    return scenario_df


def create_tableau_readme():
    """Create README for Tableau folder"""
    readme_content = """# GPU Energy Modeling Tableau Data

This directory contains CSV files prepared for use with Tableau. These files contain benchmark results and simulated power data from the GPU energy modeling framework.

## Files Overview

- `compute_benchmarks.csv` - Results from compute-intensive benchmarks (matrix multiplication, convolution)
- `memory_benchmarks.csv` - Results from memory-intensive benchmarks (memory copy, random access)
- `tbdr_benchmarks.csv` - Results from TBDR architecture benchmarks (tile memory, visibility determination, unified memory)
- `component_power.csv` - Component-level power breakdown for different workloads
- `optimization_scenarios.csv` - What-if analysis of different optimization scenarios

## Using These Files with Tableau

1. Open Tableau Desktop (or Tableau Public)
2. Click "Connect to Data" and select "Text file"
3. Navigate to this directory and select the desired CSV file
4. Use Tableau's drag-and-drop interface to create visualizations

## Sample Dashboards

The following dashboards can be created using these data files:

### 1. GPU Energy Efficiency Dashboard
- Use `compute_benchmarks.csv` and `memory_benchmarks.csv`
- Create bar charts of operations per joule for different benchmarks
- Compare energy efficiency across benchmark parameters

### 2. TBDR Architecture Analysis Dashboard
- Use `tbdr_benchmarks.csv`
- Visualize energy impact of different tile memory access patterns
- Show energy savings from visibility determination at different depth complexities

### 3. Component Power Breakdown Dashboard
- Use `component_power.csv`
- Create stacked bar charts showing power distribution across components
- Compare different workloads with pie charts of power breakdown

### 4. Optimization Impact Dashboard
- Use `optimization_scenarios.csv`
- Create what-if analysis charts showing potential energy savings
- Compare energy efficiency metrics across optimization scenarios

## Notes on Data

These files contain simulated data generated using the GPU energy modeling framework. The data is representative of the patterns and relationships you would see in real GPU power measurements, but the absolute values should not be treated as precise measurements for specific hardware.

## Learning More

For more information on using Tableau with this data:
1. Refer to the Tableau Tutorial: https://help.tableau.com/current/guides/get-started-tutorial/en-us/get-started-tutorial-home.htm
2. See the detailed explanations in the project's documentation
"""
    
    # Write README
    with open('./tableau/README.md', 'w') as f:
        f.write(readme_content)
        
    print("Created README.md in tableau directory")


def main():
    """Main function to generate all Tableau data"""
    print("Generating data for Tableau visualization...")
    
    # Ensure the tableau directory exists
    os.makedirs('./tableau', exist_ok=True)
    
    # Generate all datasets
    generate_compute_benchmark_data()
    generate_memory_benchmark_data()
    generate_tbdr_benchmark_data()
    generate_component_power_data()
    generate_optimization_scenario_data()
    
    # Create README
    create_tableau_readme()
    
    print("\nAll Tableau data files generated successfully!")
    print("You can now use these files with Tableau Desktop or Tableau Public to create visualization dashboards.")
    print("See tableau/README.md for more information on how to use these files.")


if __name__ == "__main__":
    main()