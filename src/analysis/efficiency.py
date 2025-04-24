"""
Energy efficiency analysis tools

This module provides functions for analyzing energy efficiency metrics and
identifying optimization opportunities. The methodology is based on established
energy efficiency principles from the following literature:

1. Bridges, R. A., Imam, N., & Mintz, T. M. (2016). Understanding GPU power: 
   A survey of profiling, prediction, and capping tools and strategies.
   Communications of the ACM.

2. McIntosh-Smith, S., Price, J., Deakin, T., & Poenaru, A. (2019). 
   A performance, power and energy analysis of GPU-based molecular dynamics simulations. 
   In High Performance Computing (pp. 223-242). Springer.

3. Arunkumar, A., et al. (2019). MCM-GPU: Multi-Chip-Module GPUs for continued 
   performance scaling. In 2019 ACM/IEEE 46th Annual International Symposium 
   on Computer Architecture (ISCA).

Key efficiency metrics implemented:
----------------------------------
1. Operations per watt: Throughput normalized by power consumption
2. Operations per joule: Throughput normalized by energy consumption
3. Energy per operation: Energy efficiency metric for specific operations
4. Energy-delay product (EDP): Balances energy consumption and performance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


def calculate_energy_consumption(power_data: pd.DataFrame,
                                power_col: str = 'total_power',
                                time_col: str = 'timestamp') -> float:
    """
    Calculate total energy consumption from power data
    
    Args:
        power_data: DataFrame with power measurements
        power_col: Column name for power data
        time_col: Column name for timestamp
        
    Returns:
        Total energy consumption in joules
    """
    # Calculate time deltas between measurements
    power_data = power_data.sort_values(time_col)
    time_deltas = power_data[time_col].diff().fillna(0)
    
    # Calculate energy for each interval (power * time)
    energy_increments = power_data[power_col] * time_deltas
    
    # Sum to get total energy
    total_energy = energy_increments.sum()
    
    return total_energy


def analyze_energy_efficiency(benchmark_results: Dict[str, Any],
                             power_data: Any) -> Dict[str, Any]:
    """
    Calculate and analyze energy efficiency metrics
    
    Args:
        benchmark_results: Dictionary with benchmark performance results
        power_data: DataFrame with power measurements or scalar energy value
        
    Returns:
        Dictionary with efficiency metrics
    """
    # Calculate total energy consumption
    if isinstance(power_data, pd.DataFrame):
        total_energy = calculate_energy_consumption(power_data)
        avg_power = power_data['total_power'].mean()
    else:
        # If just a scalar energy value was passed
        total_energy = float(power_data)
        avg_power = total_energy  # Default if we don't have power data
    
    # Calculate efficiency metrics
    efficiency_metrics = {}
    
    # Operations per watt (throughput per power unit)
    if 'operations' in benchmark_results:
        efficiency_metrics['operations_per_watt'] = benchmark_results['operations'] / avg_power
    
    # Operations per joule (throughput per energy unit)
    if 'operations' in benchmark_results:
        efficiency_metrics['operations_per_joule'] = benchmark_results['operations'] / total_energy
    
    # Energy per operation
    if 'operations' in benchmark_results:
        efficiency_metrics['energy_per_operation'] = total_energy / benchmark_results['operations']
    
    # Energy-delay product (EDP)
    if 'execution_time' in benchmark_results:
        execution_time = benchmark_results['execution_time']
        efficiency_metrics['energy_delay_product'] = total_energy * execution_time
    
    # Energy efficiency (operations/joule normalized to some baseline)
    if 'operations' in benchmark_results and 'baseline_operations_per_joule' in benchmark_results:
        current_ops_per_joule = benchmark_results['operations'] / total_energy
        baseline_ops_per_joule = benchmark_results['baseline_operations_per_joule']
        efficiency_metrics['relative_efficiency'] = current_ops_per_joule / baseline_ops_per_joule
    
    return efficiency_metrics


def identify_efficiency_bottlenecks(power_data: pd.DataFrame,
                                   performance_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify potential bottlenecks in energy efficiency
    
    Args:
        power_data: DataFrame with component power data
        performance_data: DataFrame with performance counter data
        
    Returns:
        Dictionary with bottleneck analysis
    """
    bottlenecks = {}
    
    # Analyze power components
    component_cols = [c for c in power_data.columns if 'power' in c and c != 'total_power']
    power_breakdown = {
        c: power_data[c].mean() for c in component_cols
    }
    
    # Find highest power component
    max_component = max(power_breakdown.items(), key=lambda x: x[1])
    bottlenecks['highest_power_component'] = {
        'component': max_component[0],
        'average_power': max_component[1],
        'percentage': max_component[1] / power_data['total_power'].mean() * 100
    }
    
    # Analyze performance counters
    if 'sm_activity' in performance_data.columns and 'memory_utilization' in performance_data.columns:
        # Check for compute vs memory bound
        avg_sm_activity = performance_data['sm_activity'].mean()
        avg_memory_util = performance_data['memory_utilization'].mean()
        
        if avg_sm_activity > 80 and avg_memory_util < 40:
            bottlenecks['workload_bottleneck'] = 'compute_bound'
        elif avg_memory_util > 80 and avg_sm_activity < 40:
            bottlenecks['workload_bottleneck'] = 'memory_bound'
        elif avg_sm_activity > 80 and avg_memory_util > 80:
            bottlenecks['workload_bottleneck'] = 'balanced_utilization'
        else:
            bottlenecks['workload_bottleneck'] = 'underutilized'
    
    # Check for power or thermal throttling
    if 'temperature' in power_data.columns:
        max_temp = power_data['temperature'].max()
        if max_temp > 85:  # Threshold for potential thermal throttling
            bottlenecks['thermal_throttling'] = {
                'max_temperature': max_temp,
                'time_above_threshold': len(power_data[power_data['temperature'] > 85]) / len(power_data) * 100
            }
    
    # Check for efficiency anomalies
    if 'total_power' in power_data.columns:
        # Calculate efficiency over time if operations data available
        if 'operations_over_time' in performance_data.columns:
            power_data['efficiency'] = performance_data['operations_over_time'] / power_data['total_power']
            efficiency_std = power_data['efficiency'].std() / power_data['efficiency'].mean()
            
            if efficiency_std > 0.2:  # High variation in efficiency
                bottlenecks['efficiency_stability'] = {
                    'coefficient_of_variation': efficiency_std,
                    'min_efficiency': power_data['efficiency'].min(),
                    'max_efficiency': power_data['efficiency'].max()
                }
    
    return bottlenecks


def what_if_analysis(model, baseline_features: np.ndarray, 
                   scenario_adjustments: Dict[str, Tuple[float, float]],
                   feature_names: List[str]) -> Dict[str, Any]:
    """
    Perform what-if analysis using the energy model
    
    Args:
        model: Trained energy model
        baseline_features: Baseline feature values
        scenario_adjustments: Dict mapping feature name to (multiplier, offset)
        feature_names: Names of features in the same order as baseline_features
        
    Returns:
        Dictionary with scenario analysis results
    """
    # Predict baseline power
    baseline_power = model.predict(baseline_features.reshape(1, -1))[0]
    
    results = {
        'baseline_power': baseline_power,
        'scenarios': {}
    }
    
    # Analyze each scenario
    for scenario_name, adjustments in scenario_adjustments.items():
        # Create adjusted feature set
        scenario_features = baseline_features.copy()
        
        # Apply adjustments to specified features
        for feature_name, (multiplier, offset) in adjustments.items():
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                scenario_features[feature_idx] = baseline_features[feature_idx] * multiplier + offset
            else:
                print(f"Warning: Feature {feature_name} not found in feature names")
        
        # Predict power for this scenario
        scenario_power = model.predict(scenario_features.reshape(1, -1))[0]
        
        # Calculate power change
        power_change = scenario_power - baseline_power
        percent_change = (power_change / baseline_power) * 100
        
        # Store results
        results['scenarios'][scenario_name] = {
            'power': scenario_power,
            'absolute_change': power_change,
            'percent_change': percent_change,
            'adjusted_features': {
                feature_names[i]: scenario_features[i] 
                for i in range(len(feature_names))
                if scenario_features[i] != baseline_features[i]
            }
        }
    
    return results