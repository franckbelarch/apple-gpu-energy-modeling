# Example Tableau Workbooks

This directory contains example Tableau workbooks (.twbx files) that demonstrate how to visualize GPU energy modeling data.

## Workbooks

- `gpu_energy_dashboard.twbx` - Complete dashboard with component power breakdown, benchmark comparisons, and optimization analysis
- `tbdr_analysis.twbx` - Focused analysis of Tile-Based Deferred Rendering architecture features
- `optimization_scenarios.twbx` - What-if analysis of different power optimization strategies

## Using These Workbooks

1. Download and install Tableau Desktop or Tableau Public
2. Open the desired .twbx file
3. Explore the pre-built visualizations
4. Use them as templates for your own analysis

## Creating Your Own Workbooks

These examples are provided as starting points. To create your own workbooks:

1. Run the data generation script: `python scripts/generate_tableau_data.py`
2. Connect to the generated CSV files in the `tableau/` directory
3. Use the provided workbooks as references for creating your visualizations
4. Refer to the `TABLEAU_GUIDE.md` file for detailed instructions

## Note

.twbx files contain both the visualization definitions and the data, making them completely self-contained for sharing.

If you're using Tableau Public, you can publish your workbooks to the Tableau Public Gallery to share your visualizations with others.