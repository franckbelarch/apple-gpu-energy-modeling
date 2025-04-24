#!/usr/bin/env python3
"""
Run benchmarks to study Apple GPU architecture patterns
"""
import os
import sys
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.compute_benchmarks import MatrixMultiplication, ConvolutionBenchmark
from src.benchmarks.memory_benchmarks import MemoryCopy, RandomAccess
from src.benchmarks.tbdr_benchmarks import TileMemoryBenchmark, VisibilityDeterminationBenchmark, UnifiedMemoryBenchmark
from src.data_collection.collectors import SimulatedPowerCollector, PerformanceCounterCollector
from src.analysis.visualization import plot_power_over_time, plot_component_breakdown
from src.analysis.efficiency import calculate_energy_consumption, analyze_energy_efficiency


def main():
    parser = argparse.ArgumentParser(description='Run Apple GPU architecture benchmarks')
    parser.add_argument('--output-dir', type=str, default='data/benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--benchmark', type=str, choices=['all', 'tile', 'visibility', 'unified', 'compute', 'memory'],
                        default='all', help='Benchmark to run')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for each benchmark')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create benchmark instances
    benchmarks = {}
    
    if args.benchmark in ['all', 'tile']:
        benchmarks['tile'] = TileMemoryBenchmark()
    
    if args.benchmark in ['all', 'visibility']:
        benchmarks['visibility'] = VisibilityDeterminationBenchmark()
    
    if args.benchmark in ['all', 'unified']:
        benchmarks['unified'] = UnifiedMemoryBenchmark()
    
    if args.benchmark in ['all', 'compute']:
        benchmarks['matmul'] = MatrixMultiplication()
        benchmarks['conv'] = ConvolutionBenchmark()
    
    if args.benchmark in ['all', 'memory']:
        benchmarks['memcopy'] = MemoryCopy()
        benchmarks['random_access'] = RandomAccess()
    
    # Create data collectors
    power_collector = SimulatedPowerCollector(output_dir=args.output_dir)
    counter_collector = PerformanceCounterCollector(output_dir=args.output_dir)
    
    # Run benchmarks
    results = {}
    power_data = {}
    counter_data = {}
    
    print("Running Apple GPU architecture benchmarks...\n")
    
    for name, benchmark in benchmarks.items():
        print(f"Running {benchmark.name} benchmark...")
        
        # Set up parameters based on benchmark type
        if name == 'tile':
            # Test different tile memory access patterns
            for pattern in ['sequential', 'random', 'alternating']:
                params = {
                    'tile_size': 32,
                    'tile_count': 100,
                    'access_pattern': pattern,
                    'overdraw': 1.0
                }
                
                # Create parameter string for naming
                param_str = f"{pattern}_pattern"
                
                print(f"  Testing {param_str}...")
                
                # Run benchmark
                result = benchmark.run(params, iterations=args.iterations)
                results[f"{name}_{param_str}"] = result
                
                # Collect simulated power data
                # Generate activity pattern that varies based on access pattern efficiency
                if pattern == 'sequential':
                    # Sequential is efficient - steady high activity
                    activity_factor = 0.7
                elif pattern == 'random':
                    # Random access is inefficient - high variability
                    activity_factor = 0.9
                else:
                    # Alternating is in between
                    activity_factor = 0.8
                
                # Create activity pattern for power simulation
                duration = 5.0  # seconds
                num_samples = int(duration / power_collector.sampling_interval)
                # Create pattern based on efficiency
                if pattern == 'sequential':
                    # Efficient pattern - steady state
                    activity_pattern = np.ones(num_samples) * activity_factor
                elif pattern == 'random':
                    # Inefficient pattern - high variability
                    activity_pattern = np.random.normal(activity_factor, 0.2, num_samples)
                    activity_pattern = np.clip(activity_pattern, 0.2, 1.0)
                else:
                    # Alternating pattern - rhythmic
                    activity_pattern = np.sin(np.linspace(0, 10, num_samples)) * 0.2 + activity_factor
                    activity_pattern = np.clip(activity_pattern, 0.2, 1.0)
                
                # Collect power data
                power_data[f"{name}_{param_str}"] = power_collector.collect_for_duration(
                    duration, activity_pattern)
                
                # Save power data
                power_df = pd.DataFrame(power_data[f"{name}_{param_str}"])
                power_df.to_csv(f"{args.output_dir}/{name}_{param_str}_power.csv", index=False)
                
        elif name == 'visibility':
            # Test different depth complexity scenarios
            for depth in [2, 5, 10]:
                params = {
                    'tile_size': 32,
                    'tile_count': 100,
                    'depth_complexity': depth,
                    'occlusion_rate': 0.7
                }
                
                # Create parameter string for naming
                param_str = f"depth_{depth}"
                
                print(f"  Testing {param_str}...")
                
                # Run benchmark
                result = benchmark.run(params, iterations=args.iterations)
                results[f"{name}_{param_str}"] = result
                
                # Collect simulated power data
                # For visibility determination, higher depth complexity means more work
                # but also more opportunity for energy savings through occlusion
                activity_factor = 0.5 + (depth / 20)  # Scale with depth complexity
                
                # Create activity pattern for power simulation
                duration = 5.0  # seconds
                num_samples = int(duration / power_collector.sampling_interval)
                
                # Create pattern - initial spike for setup, then processing
                activity_pattern = np.concatenate([
                    np.linspace(0.3, activity_factor, num_samples // 4),  # Ramp up 
                    np.ones(num_samples // 4) * activity_factor,          # Process geometry
                    np.ones(num_samples // 4) * (activity_factor * 0.7),  # HSR savings
                    np.linspace(activity_factor * 0.7, 0.3, num_samples // 4)  # Wind down
                ])
                
                # Collect power data
                power_data[f"{name}_{param_str}"] = power_collector.collect_for_duration(
                    duration, activity_pattern)
                
                # Save power data
                power_df = pd.DataFrame(power_data[f"{name}_{param_str}"])
                power_df.to_csv(f"{args.output_dir}/{name}_{param_str}_power.csv", index=False)
                
        elif name == 'unified':
            # Test different memory sharing patterns
            for pattern in ['alternating', 'producer_consumer', 'mixed_access']:
                params = {
                    'buffer_size_mb': 64,
                    'sharing_pattern': pattern,
                    'iterations': 10
                }
                
                # Create parameter string for naming
                param_str = f"{pattern}"
                
                print(f"  Testing {param_str}...")
                
                # Run benchmark
                result = benchmark.run(params, iterations=args.iterations)
                results[f"{name}_{param_str}"] = result
                
                # Collect simulated power data
                # Different sharing patterns have different power profiles
                
                # Create activity pattern for power simulation
                duration = 5.0  # seconds
                num_samples = int(duration / power_collector.sampling_interval)
                
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
                power_data[f"{name}_{param_str}"] = power_collector.collect_for_duration(
                    duration, activity_pattern)
                
                # Save power data
                power_df = pd.DataFrame(power_data[f"{name}_{param_str}"])
                power_df.to_csv(f"{args.output_dir}/{name}_{param_str}_power.csv", index=False)
        
        else:
            # Standard benchmarks use default parameters
            if name == 'matmul':
                params = {'matrix_size': 1024, 'dtype': np.float32}
            elif name == 'conv':
                params = {'input_size': 224, 'kernel_size': 3, 'channels': 3, 'filters': 64}
            elif name == 'memcopy':
                params = {'buffer_size_mb': 256, 'iterations': 5}
            elif name == 'random_access':
                params = {'array_size_mb': 128, 'access_count': 1000000}
            else:
                params = {}
            
            print(f"  Running with default parameters...")
            
            # Run benchmark
            result = benchmark.run(params, iterations=args.iterations)
            results[name] = result
            
            # Create simple activity pattern
            duration = 5.0  # seconds
            num_samples = int(duration / power_collector.sampling_interval)
            activity_pattern = np.concatenate([
                np.linspace(0.2, 0.8, num_samples // 4),  # Ramp up
                np.ones(num_samples // 2) * 0.8,          # Steady state
                np.linspace(0.8, 0.2, num_samples // 4)   # Ramp down
            ])
            
            # Collect power data
            power_data[name] = power_collector.collect_for_duration(
                duration, activity_pattern)
            
            # Save power data
            power_df = pd.DataFrame(power_data[name])
            power_df.to_csv(f"{args.output_dir}/{name}_power.csv", index=False)
    
    # Create summary report
    print("\nBenchmark Summary:")
    print("=================\n")
    
    for name, result in results.items():
        print(f"Benchmark: {name}")
        print(f"  Execution time: {result['mean_execution_time']:.4f} seconds")
        
        if 'visibility_efficiency' in result:
            print(f"  Visibility efficiency: {result['visibility_efficiency']:.2f}")
            print(f"  Estimated energy saved: {result['estimated_energy_saved']:.2f}")
        
        if 'transfers_saved' in result:
            print(f"  Transfers saved: {result['transfers_saved']:.2f}")
            print(f"  Estimated energy saved: {result['estimated_energy_saved']:.2f}")
        
        # Calculate energy consumption if we have power data
        if name in power_data:
            power_df = pd.DataFrame(power_data[name])
            energy = calculate_energy_consumption(power_df)
            print(f"  Energy consumption: {energy:.2f} joules")
            
            # If we have operations count, calculate efficiency
            if 'operations' in result:
                ops_per_joule = result['operations'] / energy
                print(f"  Operations per joule: {ops_per_joule:.2e}")
            
        print("")
    
    # Generate comparative visualizations
    print("Generating visualizations...")
    
    # Plot power for tile memory benchmarks if they were run
    if 'tile_sequential_pattern' in power_data and 'tile_random_pattern' in power_data:
        plt.figure(figsize=(12, 6))
        
        # Get dataframes
        seq_df = pd.DataFrame(power_data['tile_sequential_pattern'])
        rnd_df = pd.DataFrame(power_data['tile_random_pattern'])
        alt_df = pd.DataFrame(power_data['tile_alternating_pattern'])
        
        # Plot total power
        plt.plot(seq_df['timestamp'] - seq_df['timestamp'].min(), 
                seq_df['total_power'], label='Sequential Access')
        plt.plot(rnd_df['timestamp'] - rnd_df['timestamp'].min(), 
                rnd_df['total_power'], label='Random Access')
        plt.plot(alt_df['timestamp'] - alt_df['timestamp'].min(), 
                alt_df['total_power'], label='Alternating Access')
        
        plt.title('Power Impact of Tile Memory Access Patterns')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{args.output_dir}/tile_memory_power_comparison.png", dpi=300)
    
    # Plot power for visibility determination benchmarks if they were run
    if 'visibility_depth_2' in power_data and 'visibility_depth_10' in power_data:
        plt.figure(figsize=(12, 6))
        
        # Get dataframes
        depth2_df = pd.DataFrame(power_data['visibility_depth_2'])
        depth5_df = pd.DataFrame(power_data['visibility_depth_5'])
        depth10_df = pd.DataFrame(power_data['visibility_depth_10'])
        
        # Plot total power
        plt.plot(depth2_df['timestamp'] - depth2_df['timestamp'].min(), 
                depth2_df['total_power'], label='Depth Complexity 2')
        plt.plot(depth5_df['timestamp'] - depth5_df['timestamp'].min(), 
                depth5_df['total_power'], label='Depth Complexity 5')
        plt.plot(depth10_df['timestamp'] - depth10_df['timestamp'].min(), 
                depth10_df['total_power'], label='Depth Complexity 10')
        
        plt.title('Power Impact of Scene Depth Complexity')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{args.output_dir}/visibility_power_comparison.png", dpi=300)
    
    # Plot energy efficiency comparison if unified memory benchmarks were run
    if 'unified_alternating' in results and 'unified_producer_consumer' in results:
        # Calculate energy for each pattern
        patterns = ['alternating', 'producer_consumer', 'mixed_access']
        energy_values = []
        ops_per_joule_values = []
        
        for pattern in patterns:
            benchmark_name = f"unified_{pattern}"
            if benchmark_name in power_data:
                power_df = pd.DataFrame(power_data[benchmark_name])
                energy = calculate_energy_consumption(power_df)
                energy_values.append(energy)
                
                # Calculate operations per joule
                if 'total_operations' in results[benchmark_name]:
                    ops = results[benchmark_name]['total_operations']
                    ops_per_joule = ops / energy
                    ops_per_joule_values.append(ops_per_joule)
        
        # Plot energy comparison
        if energy_values:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(patterns, energy_values)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.title('Energy Consumption by Memory Sharing Pattern')
            plt.xlabel('Sharing Pattern')
            plt.ylabel('Energy (joules)')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.savefig(f"{args.output_dir}/unified_memory_energy_comparison.png", dpi=300)
        
        # Plot operations per joule
        if ops_per_joule_values:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(patterns, ops_per_joule_values)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2e}',
                        ha='center', va='bottom')
            
            plt.title('Energy Efficiency by Memory Sharing Pattern')
            plt.xlabel('Sharing Pattern')
            plt.ylabel('Operations per Joule')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.savefig(f"{args.output_dir}/unified_memory_efficiency_comparison.png", dpi=300)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()