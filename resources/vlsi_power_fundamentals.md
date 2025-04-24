# VLSI Power Fundamentals for GPU Architecture

## Power Consumption Basics

Total power consumption in a VLSI chip can be expressed as:

```
P_total = P_dynamic + P_static + P_short-circuit
```

Where:
- P_dynamic: Dynamic power consumption
- P_static: Static power consumption (leakage)
- P_short-circuit: Power consumption due to short-circuit currents

## Dynamic Power Consumption

Dynamic power is consumed when the circuit is actively switching states.

```
P_dynamic = α × C × V² × f
```

Where:
- α: Activity factor (fraction of gates that switch per clock cycle)
- C: Capacitance being charged/discharged
- V: Supply voltage
- f: Clock frequency

**Key points for GPU optimization:**
- Activity factor is workload-dependent and varies across different GPU components
- Modern GPUs employ techniques to minimize unnecessary switching
- Clock and power gating disable portions of the chip not in use
- Dynamic voltage and frequency scaling (DVFS) adjusts V and f based on workload

## Static Power Consumption (Leakage)

Static power is consumed even when the circuit is idle.

```
P_static = V × I_leakage
```

Main types of leakage:
1. **Subthreshold Leakage**: Current flow when transistor is "off" but not completely
2. **Gate Leakage**: Current flowing through the gate oxide
3. **Junction Leakage**: Current through reverse-biased P-N junctions

**Leakage trends in advanced nodes:**
- Leakage increases exponentially as process nodes shrink
- In modern 5nm processes (like Apple's M-series), leakage can be 30-50% of total power
- Temperature has exponential effect on leakage current
- Apple's GPU designs use advanced techniques to mitigate leakage

## Power Delivery and Distribution

**Key components:**
- **Power Delivery Network (PDN)**: Provides stable power to all parts of the chip
- **Voltage Regulators**: Convert external voltage to required chip voltage
- **Power Grids**: Metal networks that distribute power across the chip
- **Decoupling Capacitors**: Mitigate voltage fluctuations

**Challenges in GPU design:**
- Large transient currents during compute-intensive operations
- Varying load profiles across different workloads
- IR drop (voltage reduction due to resistance)
- High current density in power-hungry units

## Power Optimization Techniques

### Clock Gating

- Disables clock to inactive portions of the circuit
- Reduces dynamic power by reducing effective switching activity
- Implemented at various levels (fine-grained to coarse-grained)
- Modern GPUs have sophisticated clock gating hierarchies

Example in Apple GPUs:
```
if (tile_not_active) {
    disable_clock_to_tile_processing_units();
}
```

### Power Gating

- Cuts off power supply to inactive circuit blocks
- Eliminates both dynamic and static power in gated regions
- Requires additional isolation cells and retention strategies
- Higher overhead than clock gating (slower to wake up)

Example in Apple GPUs:
```
if (ray_tracing_units_not_needed) {
    power_gate_RT_units();
}
```

### Voltage Scaling

- Dynamically adjusts supply voltage based on performance requirements
- Quadratic impact on dynamic power (P ∝ V²)
- Multiple voltage domains allow fine-grained control
- Often combined with frequency scaling (DVFS)

Example power states in Apple GPUs:
| State | Voltage | Frequency | Power |
|-------|---------|-----------|-------|
| High Performance | 1.0V | 1.0 GHz | 100% |
| Balanced | 0.8V | 0.8 GHz | ~50% |
| Power Saving | 0.7V | 0.6 GHz | ~30% |

### Body Biasing

- Adjusts transistor threshold voltage by applying bias to substrate
- Forward body bias (FBB): Reduces threshold, increases performance and leakage
- Reverse body bias (RBB): Increases threshold, reduces leakage but also performance
- Can be applied adaptively based on workload and thermal conditions

## Thermal Considerations

- Power dissipation leads to heating
- Temperature increases leakage current exponentially
- Thermal throttling reduces performance to stay within thermal limits
- Apple's unified thermal management coordinates CPU, GPU, and Neural Engine

Temperature impact on leakage:
```
I_leakage(T) = I_leakage(T₀) × e^(k×(T-T₀))
```
Where k is a technology-dependent constant (typically 0.1-0.13 per °C)

## Power Modeling and Analysis

### Power Modeling Approaches

1. **Analytical Models**: 
   - Based on physical equations and architectural parameters
   - Typically faster but less accurate
   - Good for early design space exploration

2. **Empirical Models**:
   - Based on measured data from real hardware or detailed simulations
   - More accurate for specific implementations
   - Requires training data and statistical analysis

3. **Hybrid Models**:
   - Combine analytical foundations with empirical calibration
   - Balance accuracy and computational efficiency
   - Most commonly used in practical applications

### Component-Level Power Breakdown

Typical GPU power distribution:
- **Shader Cores**: 40-60% (highly workload dependent)
- **Memory Subsystem**: 20-35%
- **Fixed-Function Units**: 5-15%
- **Interconnect/NoC**: 5-10%
- **Clocking/PLL**: 2-5%

Example power breakdown visualization:
```
┌────────────────────────────────────────────┐
│ GPU Power Consumption                       │
│                                            │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│  │      │  │      │  │      │  │      │    │
│  │Shader│  │Memory│  │Fixed │  │Other │    │
│  │Cores │  │System│  │Func. │  │      │    │
│  │      │  │      │  │Units │  │      │    │
│  │ 50%  │  │ 30%  │  │ 10%  │  │ 10%  │    │
│  └──────┘  └──────┘  └──────┘  └──────┘    │
│                                            │
└────────────────────────────────────────────┘
```

## Advanced Power Optimization in Modern GPUs

### Architectural Optimizations

1. **Tile-Based Rendering**:
   - Processes scene in small tiles
   - Keeps active working set in on-chip memory
   - Reduces external memory bandwidth
   - Apple leverages this extensively in their GPUs

2. **Compute-to-Memory Balance**:
   - Optimizes ratio of compute units to memory bandwidth
   - Prevents power waste from stalled compute units
   - Apple's unified memory architecture helps with this balance

3. **Heterogeneous Computing**:
   - Offloads workloads to specialized hardware (ML, media engines)
   - Each specialized block is more energy efficient for its task
   - System-level scheduling optimizes overall efficiency

### Software-Hardware Co-optimization

1. **Compiler Optimizations**:
   - Instruction scheduling to minimize switching activity
   - Register allocation to reduce memory accesses
   - Loop transformations for better memory access patterns

2. **API-Level Features**:
   - Metal Performance Shaders optimize common operations
   - Memory access hints for better caching behavior
   - Power-aware shader compilation

3. **Runtime Adaptation**:
   - Dynamic workload characterization
   - Adaptive algorithm selection based on power constraints
   - Integration with system-wide power management

## Key Metrics for GPU Energy Efficiency

1. **Performance per Watt**:
   - Operations/second per watt
   - Most common overall efficiency metric

2. **Energy per Operation**:
   - Joules required per instruction or operation
   - Useful for comparing algorithms

3. **Memory Access Energy**:
   - Energy cost of different memory hierarchy levels
   - Critical for optimization (L1 vs L2 vs main memory)

4. **Energy Delay Product (EDP)**:
   - Energy × Execution Time
   - Balances power and performance

Relative energy costs in a modern GPU:
| Operation | Relative Energy Cost |
|-----------|----------------------|
| 32-bit Floating Point | 1× |
| 64-bit Floating Point | 2-4× |
| L1 Cache Access | 5-10× |
| L2 Cache Access | 20-30× |
| Main Memory Access | 100-200× |

## Resources and References

- "Power Analysis and Optimization of VLSI Circuits" by Rabaey et al.
- "Low Power Design Methodologies" by Rabaey and Pedram
- "Low Power CMOS VLSI Circuit Design" by Kaushik Roy
- "GPU Architecture: Optimization for Power Efficiency" - SIGGRAPH Course
- Apple's WWDC sessions on Metal performance optimization