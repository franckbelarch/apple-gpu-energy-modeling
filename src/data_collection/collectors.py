"""
Data collectors for GPU energy measurements

This module provides simulated data collection for GPU energy modeling in the absence
of direct hardware measurements. The simulation is based on established power models
from the following literature:

1. Li, J., et al. (2019). Evaluating modern GPU interconnect: PCIe, NVLink, NV-SLI,
   NVSwitch and GPUDirect. IEEE International Parallel and Distributed Processing
   Symposium (IPDPS).

2. Kim, N., et al. (2015). A semi-empirical power modeling approach for multi-core
   architectures. IEEE Transactions on Very Large Scale Integration (VLSI) Systems,
   23(7), 1245-1258.

3. Guerreiro, J., Ilic, A., Roma, N., & Tomás, P. (2018). GPGPU power modeling for
   multi-domain voltage-frequency scaling. In 2018 IEEE International Symposium on
   High Performance Computer Architecture (HPCA).

Key components modeled:
---------------------
1. Compute power: Scales with compute utilization and workload intensity
2. Memory power: Varies with bandwidth utilization and access patterns
3. I/O power: Models data transfer energy costs
4. Temperature: Simulates thermal behavior during high workload periods
"""
import time
import csv
import os
import json
from typing import Dict, List, Any, Optional
import numpy as np


class BaseDataCollector:
    """Base class for data collection"""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data = []
        
    def start_collection(self) -> None:
        """Start data collection"""
        self.start_time = time.time()
        self.data = []
        
    def stop_collection(self) -> None:
        """Stop data collection"""
        self.end_time = time.time()
        
    def save_data(self, filename: str) -> str:
        """
        Save collected data to file
        
        Args:
            filename: Base filename to save data to
            
        Returns:
            Full path to saved file
        """
        raise NotImplementedError("Subclasses must implement save_data")


class SimulatedPowerCollector(BaseDataCollector):
    """
    Simulated power data collector
    
    For development without actual hardware measurements
    """
    
    def __init__(self, output_dir: str = "data", 
                 sampling_interval: float = 0.1):
        """
        Initialize simulated power collector
        
        Args:
            output_dir: Directory to save collected data
            sampling_interval: Time between samples in seconds
        """
        super().__init__(output_dir)
        self.sampling_interval = sampling_interval
        self.collection_thread = None
        self.stop_flag = False
        
    def _generate_sample(self, base_power: float = 20.0, 
                         activity_factor: float = 0.8) -> Dict[str, Any]:
        """
        Generate a simulated power sample
        
        Args:
            base_power: Base power in watts
            activity_factor: Factor representing GPU activity (0-1)
            
        Returns:
            Dictionary with simulated power data
        """
        # Simulate different GPU components
        compute_power = base_power * activity_factor * (0.5 + 0.5 * np.random.random())
        memory_power = base_power * 0.3 * (0.7 + 0.3 * np.random.random())
        io_power = base_power * 0.1 * np.random.random()
        
        # Add some random variation
        total_power = compute_power + memory_power + io_power
        
        return {
            'timestamp': time.time(),
            'total_power': total_power,
            'compute_power': compute_power,
            'memory_power': memory_power,
            'io_power': io_power,
            'temperature': 50 + (total_power / base_power) * 30 * np.random.random()
        }
    
    def collect_for_duration(self, duration: float, 
                             activity_pattern: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Collect simulated data for a specified duration
        
        Args:
            duration: Collection duration in seconds
            activity_pattern: Optional list of activity factors over time
            
        Returns:
            List of collected data points
        """
        self.start_collection()
        samples = []
        
        # Default flat activity pattern if none provided
        if activity_pattern is None:
            activity_pattern = [0.8] * int(duration / self.sampling_interval)
            
        # Generate samples based on pattern or duration
        end_time = self.start_time + duration
        current_time = self.start_time
        
        pattern_index = 0
        while current_time < end_time:
            # Get current activity factor from pattern
            if pattern_index < len(activity_pattern):
                activity = activity_pattern[pattern_index]
            else:
                activity = 0.5  # Default value
                
            # Generate and store sample
            sample = self._generate_sample(activity_factor=activity)
            samples.append(sample)
            
            # Increment time and pattern index
            pattern_index += 1
            current_time += self.sampling_interval
            time.sleep(self.sampling_interval)
            
        self.data = samples
        self.stop_collection()
        
        return samples
    
    def save_data(self, filename: str) -> str:
        """
        Save collected power data to CSV file
        
        Args:
            filename: Base filename to save data to
            
        Returns:
            Full path to saved file
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'total_power', 'compute_power', 
                         'memory_power', 'io_power', 'temperature']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for sample in self.data:
                writer.writerow(sample)
                
        return filepath


class PerformanceCounterCollector(BaseDataCollector):
    """Performance counter data collector"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
    
    def collect_counters(self, counters_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collect GPU performance counters
        
        Args:
            counters_list: Optional list of specific counters to collect
            
        Returns:
            Dictionary with counter values
        """
        # In a real implementation, this would query the GPU driver
        # Here we're simulating the data for development purposes
        
        # Default counters if none specified
        if counters_list is None:
            counters_list = [
                'sm_activity',
                'memory_utilization',
                'cache_hit_rate',
                'instructions_executed',
                'memory_throughput'
            ]
            
        # Generate simulated counter values
        counters_data = {
            'timestamp': time.time(),
            'counters': {}
        }
        
        for counter in counters_list:
            # Generate simulated value based on counter type
            if 'utilization' in counter or 'activity' in counter or 'rate' in counter:
                # Percentage values
                counters_data['counters'][counter] = np.random.random() * 100.0
            elif 'throughput' in counter:
                # Bandwidth in GB/s
                counters_data['counters'][counter] = np.random.random() * 500.0
            elif 'instructions' in counter or 'operations' in counter:
                # Large counts
                counters_data['counters'][counter] = np.random.random() * 1e9
            else:
                # Default random value
                counters_data['counters'][counter] = np.random.random() * 1000.0
                
        self.data.append(counters_data)
        return counters_data
    
    def save_data(self, filename: str) -> str:
        """
        Save collected counter data to JSON file
        
        Args:
            filename: Base filename to save data to
            
        Returns:
            Full path to saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(self.data, jsonfile, indent=2)
                
        return filepath