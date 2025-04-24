"""
Visualization tools for GPU energy analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple


def plot_power_over_time(power_data: pd.DataFrame, 
                        component_cols: Optional[List[str]] = None,
                        title: str = "GPU Power Consumption Over Time",
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot power consumption over time
    
    Args:
        power_data: DataFrame with timestamp and power data
        component_cols: Optional list of component power columns
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot total power
    if 'total_power' in power_data.columns:
        ax.plot(power_data['timestamp'], power_data['total_power'], 
                label='Total Power', linewidth=2, color='black')
    
    # Plot component power if specified
    if component_cols:
        for col in component_cols:
            if col in power_data.columns:
                ax.plot(power_data['timestamp'], power_data[col], 
                        label=col.replace('_', ' ').title())
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    return fig


def plot_component_breakdown(power_data: pd.DataFrame,
                            component_cols: List[str],
                            title: str = "GPU Power Component Breakdown",
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot power breakdown by component
    
    Args:
        power_data: DataFrame with power data
        component_cols: List of component power columns
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Calculate mean values for each component
    mean_values = [power_data[col].mean() for col in component_cols]
    labels = [col.replace('_', ' ').title() for col in component_cols]
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(mean_values, labels=labels, autopct='%1.1f%%', 
           startangle=90, shadow=False)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    return fig


def plot_efficiency_comparison(benchmark_results: Dict[str, Dict[str, Any]],
                              metric: str = "operations_per_joule",
                              title: str = "Energy Efficiency Comparison",
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot energy efficiency comparison across benchmarks
    
    Args:
        benchmark_results: Dictionary of benchmark results
        metric: Efficiency metric to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    benchmark_names = []
    efficiency_values = []
    
    for name, results in benchmark_results.items():
        if metric in results:
            benchmark_names.append(name)
            efficiency_values.append(results[metric])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(benchmark_names, efficiency_values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_xlabel('Benchmark')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    return fig


def plot_model_feature_importance(feature_importance: Dict[str, float],
                                 feature_names: Optional[List[str]] = None,
                                 title: str = "Model Feature Importance",
                                 figsize: Tuple[int, int] = (12, 8),
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance from the energy model
    
    Args:
        feature_importance: Dictionary of feature importance values
        feature_names: Optional list of feature names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame for easier sorting
    if feature_names:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': list(feature_importance.values())
        })
    else:
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
    
    # Sort by absolute importance
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values('Abs_Importance', ascending=False)
    
    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
    
    # Color bars based on sign (positive/negative impact)
    for i, bar in enumerate(bars):
        if importance_df['Importance'].iloc[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
    
    ax.set_xlabel('Importance (Coefficient)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    return fig


def create_power_heatmap(activity_data: pd.DataFrame,
                        power_data: pd.DataFrame,
                        x_metric: str,
                        y_metric: str,
                        power_metric: str = 'total_power',
                        title: str = "Power Consumption Heatmap",
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap of power consumption vs. two activity metrics
    
    Args:
        activity_data: DataFrame with workload activity metrics
        power_data: DataFrame with power measurements
        x_metric: Column name for x-axis
        y_metric: Column name for y-axis
        power_metric: Column name for power measurement
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Merge data if timestamps align
    if 'timestamp' in activity_data.columns and 'timestamp' in power_data.columns:
        # Create bins for the metrics
        x_bins = np.linspace(activity_data[x_metric].min(), 
                            activity_data[x_metric].max(), 10)
        y_bins = np.linspace(activity_data[y_metric].min(), 
                            activity_data[y_metric].max(), 10)
        
        # Assign each data point to a bin
        activity_data['x_bin'] = pd.cut(activity_data[x_metric], bins=x_bins, labels=False)
        activity_data['y_bin'] = pd.cut(activity_data[y_metric], bins=y_bins, labels=False)
        
        # Merge with power data based on closest timestamp
        merged_data = pd.merge_asof(activity_data.sort_values('timestamp'), 
                                  power_data.sort_values('timestamp'),
                                  on='timestamp')
        
        # Create pivot table for heatmap
        pivot_data = merged_data.pivot_table(
            index='y_bin', 
            columns='x_bin',
            values=power_metric,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax)
        
        # Set labels
        x_bin_centers = [(x_bins[i] + x_bins[i+1])/2 for i in range(len(x_bins)-1)]
        y_bin_centers = [(y_bins[i] + y_bins[i+1])/2 for i in range(len(y_bins)-1)]
        
        ax.set_xticklabels([f'{x:.1f}' for x in x_bin_centers], rotation=45)
        ax.set_yticklabels([f'{y:.1f}' for y in y_bin_centers])
        
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig
    else:
        raise ValueError("Both activity_data and power_data must have timestamp column")