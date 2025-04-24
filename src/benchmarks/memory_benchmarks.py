"""
Memory-intensive benchmarks for GPU energy modeling
"""
import numpy as np
from typing import Dict, Any
from .base import GPUBenchmark


class MemoryCopy(GPUBenchmark):
    """Memory copy benchmark to evaluate memory bandwidth impacts on energy"""
    
    def __init__(self):
        super().__init__(
            name="memory_copy",
            description="Memory copy operations of configurable size",
            category="memory"
        )
    
    def _execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory copy with given parameters"""
        # Extract parameters with defaults
        buffer_size_mb = parameters.get('buffer_size_mb', 1024)  # Size in MB
        iterations = parameters.get('iterations', 10)
        dtype = parameters.get('dtype', np.float32)
        
        # Calculate number of elements based on buffer size
        bytes_per_element = np.dtype(dtype).itemsize
        elements = int((buffer_size_mb * 1024 * 1024) / bytes_per_element)
        
        # Create source buffer
        source = np.random.random(elements).astype(dtype)
        
        # Perform memory copies
        total_copied = 0
        for _ in range(iterations):
            # In a real implementation, this would use GPU memory operations
            # through CUDA or equivalent
            destination = np.copy(source)
            total_copied += destination.nbytes
        
        return {
            'result': float(np.sum(destination)),  # Simple checksum
            'memory_copied': total_copied,
            'memory_bandwidth': total_copied / (iterations * parameters.get('estimated_time', 1.0))
        }


class RandomAccess(GPUBenchmark):
    """Random memory access patterns to evaluate memory hierarchy impacts"""
    
    def __init__(self):
        super().__init__(
            name="random_access",
            description="Random memory access patterns",
            category="memory"
        )
    
    def _execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute random access benchmark with given parameters"""
        # Extract parameters with defaults
        array_size_mb = parameters.get('array_size_mb', 256)  # Size in MB
        access_count = parameters.get('access_count', 10000000)
        dtype = parameters.get('dtype', np.float32)
        
        # Calculate number of elements based on array size
        bytes_per_element = np.dtype(dtype).itemsize
        elements = int((array_size_mb * 1024 * 1024) / bytes_per_element)
        
        # Create data array
        data_array = np.random.random(elements).astype(dtype)
        
        # Generate random access indices
        indices = np.random.randint(0, elements, access_count)
        
        # Perform random accesses
        # In a real implementation, this would use GPU memory operations
        result = 0.0
        for idx in indices:
            result += data_array[idx]
        
        return {
            'result': float(result),
            'memory_accessed': access_count * bytes_per_element,
            'access_pattern': 'random',
            'array_size': data_array.nbytes
        }