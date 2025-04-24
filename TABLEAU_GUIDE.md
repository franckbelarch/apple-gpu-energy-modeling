# Using Tableau with GPU Energy Modeling

This guide explains how to use Tableau with the GPU energy modeling project to create interactive visualizations and dashboards.

## Overview

Tableau is a powerful data visualization tool that can help analyze and communicate insights from GPU power and performance data. This project includes scripts to generate CSV files specifically formatted for use with Tableau, allowing you to create professional-quality dashboards to explore energy efficiency patterns.

## Prerequisites

1. **Tableau Desktop/Public**: 
   - Tableau Desktop (commercial software)
   - Tableau Public (free version): https://public.tableau.com/

2. **Data Generation**:
   - Run the script to generate Tableau-ready data:
     ```bash
     python scripts/generate_tableau_data.py
     ```
   - This creates CSV files in the `tableau/` directory

## Creating Basic Visualizations

### 1. GPU Component Power Breakdown

This visualization shows how power is distributed across different GPU components.

1. Open Tableau and connect to `tableau/component_power.csv`
2. Drag 'workload_description' to Columns
3. Drag 'compute_power', 'memory_power', and 'io_power' to Rows
4. Go to 'Show Me' and select the stacked bar chart option
5. For a percentage view:
   - Change the visualization to 100% stacked bars
   - Use the calculated fields 'compute_percent', 'memory_percent', and 'io_percent'

### 2. Benchmark Performance/Watt Comparison

This visualization compares energy efficiency across different benchmarks.

1. Connect to `tableau/compute_benchmarks.csv` or `tableau/memory_benchmarks.csv`
2. Drag 'parameter' to Columns
3. Drag 'operations_per_joule' to Rows
4. Split by 'benchmark' using color
5. Add labels to show the exact values

### 3. TBDR Architecture Energy Benefits

This visualization demonstrates the energy benefits of TBDR architecture features.

1. Connect to `tableau/tbdr_benchmarks.csv`
2. Create a new worksheet for Tile Memory patterns:
   - Drag 'access_pattern' to Columns
   - Drag 'energy_consumption' to Rows
   - Format as a bar chart
3. Create a second worksheet for Visibility Determination:
   - Drag 'depth_complexity' to Columns
   - Drag 'estimated_energy_saved' to Rows
   - Format as a bar chart

### 4. Optimization Impact Analysis

This visualization shows the potential impact of different optimization strategies.

1. Connect to `tableau/optimization_scenarios.csv`
2. Drag 'scenario_description' to Columns
3. Drag 'power_reduction_percent' to Rows
4. Use 'workload' on the Color mark
5. Add a reference line at 0% to highlight the baseline

## Creating a Dashboard

Combine your visualizations into a comprehensive dashboard:

1. Create a new dashboard
2. Add the visualizations you created earlier
3. Arrange them in a logical layout
4. Add titles, text explanations, and filters
5. Consider adding interactive elements:
   - Parameter controls to adjust visualization parameters
   - Filter actions to allow clicking on one visualization to filter others
   - Highlight actions to see related data across visualizations

## Example Dashboard Layout

Here's a suggested layout for a GPU Energy Analysis Dashboard:

```
+---------------------------------------+
| GPU Energy Analysis Dashboard         |
+---------------+----------------------+
| Component     | Performance/Watt     |
| Power         | Comparison           |
| Breakdown     |                      |
|               |                      |
+---------------+----------------------+
| TBDR Architecture Benefits           |
|                                      |
+---------------+----------------------+
| Optimization  | Efficiency           |
| Impact        | Metrics              |
|               |                      |
+---------------+----------------------+
| Interactive Controls & Filters       |
+--------------------------------------+
```

## Advanced Techniques

### 1. Creating Calculated Fields

You can create calculated fields for more advanced analysis:

1. Right-click in the Data pane and select "Create Calculated Field"
2. Example calculations:
   - Energy efficiency ratio: `[operations_per_joule] / [Baseline operations_per_joule]`
   - Power savings percentage: `([baseline_power] - [total_power]) / [baseline_power] * 100`
   - Performance per watt ratio: `[performance] / [total_power]`

### 2. Trend Analysis

For time-series data:

1. Connect to power over time data
2. Use trend lines to identify patterns
3. Create moving averages to smooth noisy data
4. Use forecast functionality to predict future values

### 3. What-If Analysis

Create interactive parameters to allow users to explore scenarios:

1. Create a parameter (e.g., "Compute Power Reduction")
2. Create a calculated field that uses this parameter
   - Example: `[compute_power] * (1 - [Compute Power Reduction])`
3. Use this field in visualizations
4. Add the parameter control to the dashboard

## Publishing and Sharing

### With Tableau Public

1. Save your workbook
2. Click "File" > "Save to Tableau Public"
3. Create or sign in to your Tableau Public account
4. Your dashboard will be published to the web where you can share the URL

### With Tableau Desktop

1. Save your workbook as a Tableau Packaged Workbook (.twbx)
2. Share the .twbx file, which includes both the visualizations and the data
3. Recipients with Tableau Desktop or Tableau Reader can open the file

## Additional Resources

- [Tableau Tutorials](https://help.tableau.com/current/guides/get-started-tutorial/en-us/get-started-tutorial-home.htm)
- [Tableau Public Gallery](https://public.tableau.com/gallery) for inspiration
- [GPU Energy Modeling Documentation](./VERIFICATION.md) for understanding the data

## Example Tableau Workbook

An example Tableau workbook (compatible with Tableau 2021.4 or later) is available in the `tableau/examples/` directory. Open this file with Tableau to see pre-built visualizations that you can use as starting points.