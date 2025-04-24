"""
Optimization identification system for GPU energy efficiency

This module provides tools to identify power optimization opportunities
based on performance counter and power data analysis.

References:
1. Wang, K., et al. (2019). "Energy-aware GPU power profiling and prediction for 
   efficient job scheduling." Journal of Parallel and Distributed Computing.
2. Li, D., et al. (2015). "Orchestrating Heterogeneous Resources for Improving Energy 
   Efficiency of Mobile Computing." ACM Transactions on Embedded Computing Systems.
3. Kim, J., et al. (2020). "AutoScale: Energy Efficiency Optimization for Stochastic 
   Edge Inference Using Reinforcement Learning." IEEE International Symposium on
   Workload Characterization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def identify_hotspots(power_data: pd.DataFrame,
                     counter_data: pd.DataFrame,
                     power_threshold_percentage: float = 0.7) -> Dict[str, Any]:
    """
    Identify time periods with high power consumption (hotspots)
    
    Args:
        power_data: DataFrame with power measurements
        counter_data: DataFrame with performance counter data
        power_threshold_percentage: Power threshold as percentage of max power
        
    Returns:
        Dictionary with hotspot information
    """
    # Calculate power threshold
    max_power = power_data['total_power'].max()
    power_threshold = max_power * power_threshold_percentage
    
    # Identify hotspots
    hotspots = power_data[power_data['total_power'] >= power_threshold]
    
    # If no hotspots found, return empty results
    if len(hotspots) == 0:
        return {
            'hotspots_found': False,
            'message': f"No periods exceed {power_threshold_percentage*100:.0f}% of max power ({max_power:.2f}W)"
        }
    
    # Calculate hotspot statistics
    hotspot_stats = {
        'hotspots_found': True,
        'count': len(hotspots),
        'percentage_of_time': len(hotspots) / len(power_data) * 100,
        'avg_power': hotspots['total_power'].mean(),
        'max_power': hotspots['total_power'].max(),
        'power_threshold': power_threshold,
        'total_energy_percentage': hotspots['total_power'].sum() / power_data['total_power'].sum() * 100,
        'hotspot_periods': []
    }
    
    # Find continuous periods of hotspots
    hotspot_indices = hotspots.index.tolist()
    periods = []
    start_idx = hotspot_indices[0]
    
    for i in range(1, len(hotspot_indices)):
        if hotspot_indices[i] != hotspot_indices[i-1] + 1:
            # End of continuous period
            periods.append((start_idx, hotspot_indices[i-1]))
            start_idx = hotspot_indices[i]
    
    # Add the last period
    periods.append((start_idx, hotspot_indices[-1]))
    
    # Analyze each hotspot period
    for start, end in periods:
        period_data = power_data.loc[start:end]
        
        # Get corresponding counter data if timestamps align
        period_counters = None
        if 'timestamp' in counter_data.columns and 'timestamp' in period_data.columns:
            # Find counters with timestamps in this period
            start_time = period_data['timestamp'].min()
            end_time = period_data['timestamp'].max()
            period_counters = counter_data[
                (counter_data['timestamp'] >= start_time) & 
                (counter_data['timestamp'] <= end_time)
            ]
        
        # Calculate period statistics
        period_stats = {
            'start_index': start,
            'end_index': end,
            'duration': end - start,
            'avg_power': period_data['total_power'].mean(),
            'max_power': period_data['total_power'].max(),
            'energy_consumption': period_data['total_power'].sum(),
            'dominant_component': _get_dominant_component(period_data)
        }
        
        # Add counter insights if available
        if period_counters is not None:
            period_stats['counter_insights'] = _analyze_counters(period_counters)
        
        hotspot_stats['hotspot_periods'].append(period_stats)
    
    return hotspot_stats


def _get_dominant_component(power_data: pd.DataFrame) -> str:
    """
    Determine the dominant power component in the data
    
    Args:
        power_data: DataFrame with component power columns
        
    Returns:
        Name of dominant component
    """
    # Look for component power columns
    component_cols = [col for col in power_data.columns 
                     if col.endswith('_power') and col != 'total_power']
    
    if not component_cols:
        return "unknown"
    
    # Calculate average power for each component
    avg_powers = {col: power_data[col].mean() for col in component_cols}
    
    # Find dominant component
    dominant_component = max(avg_powers.items(), key=lambda x: x[1])
    
    return dominant_component[0]


def _analyze_counters(counter_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze performance counters to identify potential issues
    
    Args:
        counter_data: DataFrame with performance counter data
        
    Returns:
        Dictionary with counter insights
    """
    insights = {}
    
    # Check for memory-related counters
    memory_counters = [col for col in counter_data.columns 
                      if 'memory' in col.lower() or 'cache' in col.lower()]
    
    if memory_counters:
        # Check for high memory utilization
        for counter in memory_counters:
            if 'utilization' in counter.lower() and counter_data[counter].mean() > 80:
                insights['high_memory_utilization'] = True
            if 'hit_rate' in counter.lower() and counter_data[counter].mean() < 50:
                insights['low_cache_hit_rate'] = True
    
    # Check for compute-related counters
    compute_counters = [col for col in counter_data.columns 
                       if 'sm' in col.lower() or 'compute' in col.lower()]
    
    if compute_counters:
        # Check for high compute utilization
        for counter in compute_counters:
            if 'utilization' in counter.lower() and counter_data[counter].mean() > 80:
                insights['high_compute_utilization'] = True
    
    # Check for underutilization (opportunity for DVFS)
    utilization_counters = [col for col in counter_data.columns if 'utilization' in col.lower()]
    if utilization_counters:
        avg_utilization = np.mean([counter_data[col].mean() for col in utilization_counters])
        if avg_utilization < 40:
            insights['underutilized'] = True
    
    return insights


def identify_inefficient_patterns(counter_data: pd.DataFrame, 
                                 power_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify inefficient patterns in GPU usage using counter data
    
    Args:
        counter_data: DataFrame with performance counter data
        power_data: DataFrame with power measurements
        
    Returns:
        Dictionary with inefficiency patterns
    """
    patterns = {}
    
    # Merge counters and power if timestamps are available
    merged_data = None
    if 'timestamp' in counter_data.columns and 'timestamp' in power_data.columns:
        counter_data = counter_data.sort_values('timestamp')
        power_data = power_data.sort_values('timestamp')
        
        # Merge on closest timestamp
        merged_data = pd.merge_asof(counter_data, power_data, on='timestamp')
    
    # Look for pattern 1: High memory utilization with low compute utilization
    if 'memory_utilization' in counter_data.columns and 'sm_activity' in counter_data.columns:
        is_memory_bound = (counter_data['memory_utilization'] > 80) & (counter_data['sm_activity'] < 40)
        memory_bound_percentage = is_memory_bound.mean() * 100
        
        if memory_bound_percentage > 10:  # At least 10% of time is memory-bound
            patterns['memory_bound'] = {
                'percentage': memory_bound_percentage,
                'avg_memory_utilization': counter_data.loc[is_memory_bound, 'memory_utilization'].mean(),
                'avg_compute_utilization': counter_data.loc[is_memory_bound, 'sm_activity'].mean(),
                'recommendation': "Consider memory access pattern optimization"
            }
    
    # Look for pattern 2: High compute utilization with high power but low ops/watt
    if merged_data is not None and 'sm_activity' in merged_data.columns:
        high_compute = merged_data['sm_activity'] > 80
        
        if high_compute.any():
            high_compute_data = merged_data[high_compute]
            
            if 'operations' in high_compute_data.columns and 'total_power' in high_compute_data.columns:
                # Calculate ops/watt
                high_compute_data['ops_per_watt'] = high_compute_data['operations'] / high_compute_data['total_power']
                
                # Check if ops/watt is lower than average
                all_ops_per_watt = merged_data['operations'] / merged_data['total_power']
                
                if high_compute_data['ops_per_watt'].mean() < all_ops_per_watt.mean() * 0.8:
                    patterns['inefficient_compute'] = {
                        'percentage': high_compute.mean() * 100,
                        'avg_ops_per_watt': high_compute_data['ops_per_watt'].mean(),
                        'baseline_ops_per_watt': all_ops_per_watt.mean(),
                        'efficiency_ratio': high_compute_data['ops_per_watt'].mean() / all_ops_per_watt.mean(),
                        'recommendation': "Look for compute-intensive code sections with poor efficiency"
                    }
    
    # Look for pattern 3: Low utilization overall but high power draw (idle power issue)
    if merged_data is not None:
        utilization_cols = [col for col in merged_data.columns if 'utilization' in col.lower() or 'activity' in col.lower()]
        
        if utilization_cols:
            # Calculate average utilization across all utilization metrics
            merged_data['avg_utilization'] = merged_data[utilization_cols].mean(axis=1)
            
            # Identify low utilization periods
            low_util = merged_data['avg_utilization'] < 30
            
            if low_util.any():
                low_util_data = merged_data[low_util]
                
                # Check if power consumption is still significant during low utilization
                if 'total_power' in low_util_data.columns:
                    avg_low_util_power = low_util_data['total_power'].mean()
                    max_power = merged_data['total_power'].max()
                    
                    if avg_low_util_power > max_power * 0.4:  # Power is still >40% of max during low utilization
                        patterns['high_idle_power'] = {
                            'percentage': low_util.mean() * 100,
                            'avg_utilization': low_util_data['avg_utilization'].mean(),
                            'avg_power': avg_low_util_power,
                            'power_percentage': avg_low_util_power / max_power * 100,
                            'recommendation': "Investigate idle power consumption and power gating options"
                        }
    
    # Look for pattern 4: Frequent power spikes (potential for smoothing)
    if 'total_power' in power_data.columns:
        # Calculate power deltas between adjacent measurements
        power_data['power_delta'] = power_data['total_power'].diff()
        
        # Calculate statistics on absolute power changes
        abs_deltas = power_data['power_delta'].abs()
        avg_delta = abs_deltas.mean()
        max_delta = abs_deltas.max()
        power_variance = power_data['total_power'].var()
        
        # Check for spiky pattern
        if power_variance > (power_data['total_power'].mean() * 0.3)**2:
            patterns['power_spikes'] = {
                'avg_power_change': avg_delta,
                'max_power_change': max_delta,
                'power_variance': power_variance,
                'recommendation': "Look for bursty workloads that could benefit from smoothing"
            }
    
    return patterns


def identify_dvfs_opportunities(counter_data: pd.DataFrame,
                              power_data: pd.DataFrame,
                              utilization_threshold: float = 0.4) -> Dict[str, Any]:
    """
    Identify opportunities for dynamic voltage and frequency scaling
    
    Args:
        counter_data: DataFrame with performance counter data
        power_data: DataFrame with power measurements
        utilization_threshold: Threshold below which DVFS can be applied
        
    Returns:
        Dictionary with DVFS recommendations
    """
    dvfs_opportunities = {}
    
    # Check for utilization data
    utilization_cols = [col for col in counter_data.columns 
                       if 'utilization' in col.lower() or 'activity' in col.lower()]
    
    if not utilization_cols:
        return {
            'error': "No utilization data found in counter data",
            'dvfs_potential': 'unknown'
        }
    
    # Calculate overall utilization
    counter_data['avg_utilization'] = counter_data[utilization_cols].mean(axis=1)
    
    # Identify periods with low utilization
    low_util = counter_data['avg_utilization'] < utilization_threshold * 100  # Assuming percentages
    low_util_percentage = low_util.mean() * 100
    
    if low_util_percentage < 5:
        dvfs_opportunities['dvfs_potential'] = 'minimal'
        dvfs_opportunities['low_util_percentage'] = low_util_percentage
        dvfs_opportunities['recommendation'] = "Workload has consistently high utilization, little opportunity for DVFS"
        return dvfs_opportunities
    
    # Cluster the utilization data to find distinct operational modes
    # This helps identify when DVFS might be applied
    if len(counter_data) > 10:  # Need enough data for clustering
        # Extract relevant features for clustering
        features = counter_data[utilization_cols].copy()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters (simple heuristic)
        max_clusters = min(5, len(counter_data) // 10)
        inertia = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertia.append(kmeans.inertia_)
        
        # Simple elbow method to find optimal clusters
        optimal_clusters = 2  # Default
        for i in range(1, len(inertia) - 1):
            prev_reduction = inertia[i-1] - inertia[i]
            next_reduction = inertia[i] - inertia[i+1]
            if next_reduction / prev_reduction < 0.5:  # Diminishing returns
                optimal_clusters = i + 1
                break
        
        # Cluster the data
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        counter_data['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Analyze each cluster
        cluster_analysis = []
        
        for cluster in range(optimal_clusters):
            cluster_data = counter_data[counter_data['cluster'] == cluster]
            
            # Calculate mean utilization for this cluster
            mean_util = cluster_data['avg_utilization'].mean()
            
            # Determine if this cluster is a DVFS opportunity
            is_dvfs_opportunity = mean_util < utilization_threshold * 100
            
            # Calculate percentage of time in this cluster
            percentage = len(cluster_data) / len(counter_data) * 100
            
            # Estimate potential savings
            potential_savings = 0.0
            if is_dvfs_opportunity and 'total_power' in power_data.columns:
                # Merge counter data with power data if possible
                if 'timestamp' in counter_data.columns and 'timestamp' in power_data.columns:
                    # Get timestamps for this cluster
                    cluster_times = cluster_data['timestamp'].values
                    
                    # Find closest power measurements
                    cluster_power = []
                    for t in cluster_times:
                        # Find closest timestamp
                        closest_idx = np.abs(power_data['timestamp'].values - t).argmin()
                        cluster_power.append(power_data.iloc[closest_idx]['total_power'])
                    
                    if cluster_power:
                        avg_power = np.mean(cluster_power)
                        
                        # Estimate savings based on frequency-voltage relationship
                        # P ~ V² * f, and V ~ f, so P ~ f³
                        # Here we use a simplified model: if utilization is at 40%,
                        # we could reduce frequency to 70% and save about 50% power
                        freq_reduction = max(0.7, mean_util / 100)  # Don't go below 70% frequency
                        estimated_power = avg_power * (freq_reduction ** 3)
                        potential_savings = (avg_power - estimated_power) / avg_power * 100
            
            cluster_analysis.append({
                'cluster': cluster,
                'mean_utilization': mean_util,
                'is_dvfs_opportunity': is_dvfs_opportunity,
                'percentage_of_time': percentage,
                'potential_power_savings': potential_savings
            })
        
        # Sort clusters by potential savings
        cluster_analysis.sort(key=lambda x: x['potential_power_savings'], reverse=True)
        
        # Calculate overall potential savings
        total_savings = sum(c['potential_power_savings'] * c['percentage_of_time'] / 100 
                          for c in cluster_analysis)
        
        dvfs_opportunities['dvfs_potential'] = 'significant' if total_savings > 15 else 'moderate'
        dvfs_opportunities['estimated_power_savings'] = total_savings
        dvfs_opportunities['cluster_analysis'] = cluster_analysis
        
        # Generate recommendations
        recommendations = []
        for cluster in cluster_analysis:
            if cluster['is_dvfs_opportunity'] and cluster['percentage_of_time'] > 5:
                recommendations.append(
                    f"Reduce frequency during operational mode with {cluster['mean_utilization']:.1f}% "
                    f"utilization ({cluster['percentage_of_time']:.1f}% of time) for potential "
                    f"{cluster['potential_power_savings']:.1f}% power savings in this mode"
                )
        
        dvfs_opportunities['recommendations'] = recommendations
    
    else:
        # Not enough data for clustering
        dvfs_opportunities['dvfs_potential'] = 'unknown'
        dvfs_opportunities['error'] = "Insufficient data for DVFS analysis"
    
    return dvfs_opportunities


def generate_optimization_recommendations(hotspots: Dict[str, Any],
                                        patterns: Dict[str, Any],
                                        dvfs_opportunities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate prioritized optimization recommendations
    
    Args:
        hotspots: Results from identify_hotspots
        patterns: Results from identify_inefficient_patterns
        dvfs_opportunities: Results from identify_dvfs_opportunities
        
    Returns:
        List of recommendations sorted by estimated impact
    """
    recommendations = []
    
    # Hotspot-based recommendations
    if hotspots.get('hotspots_found', False):
        # Look at dominant components in hotspots
        for period in hotspots.get('hotspot_periods', []):
            dominant = period.get('dominant_component', '')
            
            if 'compute' in dominant:
                recommendations.append({
                    'category': 'hotspot',
                    'component': 'compute',
                    'description': f"Optimize compute-intensive section at period {period['start_index']}-{period['end_index']}",
                    'estimated_impact': 'high',
                    'estimated_savings': period['energy_consumption'] / hotspots['total_energy_percentage'] * 0.3  # Assume 30% reduction
                })
            elif 'memory' in dominant:
                recommendations.append({
                    'category': 'hotspot',
                    'component': 'memory',
                    'description': f"Optimize memory access patterns at period {period['start_index']}-{period['end_index']}",
                    'estimated_impact': 'high',
                    'estimated_savings': period['energy_consumption'] / hotspots['total_energy_percentage'] * 0.4  # Assume 40% reduction
                })
            elif 'io' in dominant:
                recommendations.append({
                    'category': 'hotspot',
                    'component': 'io',
                    'description': f"Optimize I/O operations at period {period['start_index']}-{period['end_index']}",
                    'estimated_impact': 'medium',
                    'estimated_savings': period['energy_consumption'] / hotspots['total_energy_percentage'] * 0.2  # Assume 20% reduction
                })
    
    # Pattern-based recommendations
    for pattern, details in patterns.items():
        if pattern == 'memory_bound':
            recommendations.append({
                'category': 'pattern',
                'component': 'memory',
                'description': "Optimize memory access patterns to reduce memory-bound execution",
                'estimated_impact': 'high',
                'recommendation': details.get('recommendation', ''),
                'estimated_savings': details.get('percentage', 0) * 0.4 / 100  # Assume 40% improvement
            })
        elif pattern == 'inefficient_compute':
            recommendations.append({
                'category': 'pattern',
                'component': 'compute',
                'description': "Identify and optimize compute operations with low efficiency",
                'estimated_impact': 'medium',
                'recommendation': details.get('recommendation', ''),
                'estimated_savings': (1 - details.get('efficiency_ratio', 0.5)) * details.get('percentage', 0) / 100
            })
        elif pattern == 'high_idle_power':
            recommendations.append({
                'category': 'pattern',
                'component': 'power_management',
                'description': "Implement better power gating during low utilization periods",
                'estimated_impact': 'medium',
                'recommendation': details.get('recommendation', ''),
                'estimated_savings': details.get('percentage', 0) * details.get('power_percentage', 0) * 0.6 / 10000  # Complex calculation
            })
        elif pattern == 'power_spikes':
            recommendations.append({
                'category': 'pattern',
                'component': 'workload_scheduling',
                'description': "Smooth workload to reduce power spikes and improve efficiency",
                'estimated_impact': 'low',
                'recommendation': details.get('recommendation', ''),
                'estimated_savings': 0.05  # Assume 5% overall improvement
            })
    
    # DVFS recommendations
    if dvfs_opportunities.get('dvfs_potential') in ['significant', 'moderate']:
        for i, rec in enumerate(dvfs_opportunities.get('recommendations', [])):
            impact = 'high' if i == 0 else 'medium'  # First recommendation is highest impact
            recommendations.append({
                'category': 'dvfs',
                'component': 'frequency_scaling',
                'description': rec,
                'estimated_impact': impact,
                'estimated_savings': dvfs_opportunities.get('estimated_power_savings', 10) / 100
            })
    
    # Sort recommendations by estimated impact and savings
    impact_scores = {'high': 3, 'medium': 2, 'low': 1}
    for rec in recommendations:
        rec['impact_score'] = impact_scores.get(rec['estimated_impact'], 0) * rec.get('estimated_savings', 0)
    
    recommendations.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
    
    return recommendations


def visualize_hotspots(power_data: pd.DataFrame, 
                      hotspots: Dict[str, Any],
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize power hotspots
    
    Args:
        power_data: DataFrame with power measurements
        hotspots: Results from identify_hotspots
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot total power
    ax.plot(power_data.index, power_data['total_power'], label='Total Power', color='blue', alpha=0.7)
    
    # Highlight hotspot periods
    if hotspots.get('hotspots_found', False):
        for period in hotspots.get('hotspot_periods', []):
            start, end = period['start_index'], period['end_index']
            ax.axvspan(start, end, alpha=0.3, color='red')
            
            # Label the hotspot
            mid_point = (start + end) // 2
            ax.text(mid_point, power_data['total_power'].max() * 0.9, f"Hotspot", 
                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add threshold line
    if hotspots.get('power_threshold'):
        ax.axhline(y=hotspots['power_threshold'], color='red', linestyle='--', 
                  label=f"Threshold ({hotspots['power_threshold']:.2f}W)")
    
    # Format plot
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Consumption Hotspots')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig


def visualize_optimization_impact(recommendations: List[Dict[str, Any]],
                                figsize: Tuple[int, int] = (12, 8),
                                max_recommendations: int = 5,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize estimated impact of optimization recommendations
    
    Args:
        recommendations: Results from generate_optimization_recommendations
        figsize: Figure size
        max_recommendations: Maximum number of recommendations to show
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Limit to top recommendations
    if len(recommendations) > max_recommendations:
        recommendations = recommendations[:max_recommendations]
    
    # Extract data
    descriptions = [rec.get('description', f"Recommendation {i+1}") 
                   for i, rec in enumerate(recommendations)]
    savings = [rec.get('estimated_savings', 0) * 100 for rec in recommendations]  # Convert to percentage
    categories = [rec.get('category', 'unknown') for rec in recommendations]
    components = [rec.get('component', 'unknown') for rec in recommendations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    bars = ax.barh(descriptions, savings)
    
    # Color bars by category
    category_colors = {'hotspot': 'crimson', 'pattern': 'royalblue', 'dvfs': 'darkgreen'}
    for i, bar in enumerate(bars):
        bar.set_color(category_colors.get(categories[i], 'gray'))
    
    # Add labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
               f"{savings[i]:.1f}%", va='center')
    
    # Add component labels
    for i, (bar, component) in enumerate(zip(bars, components)):
        ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2, 
               component, va='center', ha='center', color='white', fontweight='bold')
    
    # Format plot
    ax.set_xlabel('Estimated Power Reduction (%)')
    ax.set_title('Estimated Impact of Optimization Recommendations')
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat.capitalize()) 
                      for cat, color in category_colors.items() 
                      if cat in categories]
    ax.legend(handles=legend_elements, loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig


def generate_optimization_report(hotspots: Dict[str, Any],
                               patterns: Dict[str, Any],
                               dvfs_opportunities: Dict[str, Any],
                               recommendations: List[Dict[str, Any]],
                               output_format: str = 'text') -> str:
    """
    Generate a comprehensive optimization report
    
    Args:
        hotspots: Results from identify_hotspots
        patterns: Results from identify_inefficient_patterns
        dvfs_opportunities: Results from identify_dvfs_opportunities
        recommendations: Results from generate_optimization_recommendations
        output_format: Format of the report ('text' or 'html')
        
    Returns:
        Report as text or HTML
    """
    if output_format == 'html':
        report = "<h1>GPU Energy Optimization Report</h1>\n"
        
        # Hotspots section
        report += "<h2>Power Hotspots</h2>\n"
        if hotspots.get('hotspots_found', False):
            report += f"<p>{hotspots['count']} hotspots found, consuming {hotspots['total_energy_percentage']:.1f}% of total energy</p>\n"
            report += "<ul>\n"
            for period in hotspots.get('hotspot_periods', []):
                report += f"<li>Hotspot at period {period['start_index']}-{period['end_index']}, "
                report += f"avg power: {period['avg_power']:.2f}W, "
                report += f"dominant component: {period['dominant_component']}</li>\n"
            report += "</ul>\n"
        else:
            report += "<p>No significant power hotspots found.</p>\n"
        
        # Inefficient patterns section
        report += "<h2>Inefficient Patterns</h2>\n"
        if patterns:
            report += "<ul>\n"
            for pattern, details in patterns.items():
                report += f"<li><strong>{pattern.replace('_', ' ').title()}</strong>: "
                if 'recommendation' in details:
                    report += f"{details['recommendation']}</li>\n"
                else:
                    report += "Inefficient pattern detected</li>\n"
            report += "</ul>\n"
        else:
            report += "<p>No significant inefficient patterns detected.</p>\n"
        
        # DVFS opportunities section
        report += "<h2>DVFS Opportunities</h2>\n"
        if dvfs_opportunities.get('dvfs_potential') in ['significant', 'moderate']:
            report += f"<p>DVFS potential: {dvfs_opportunities['dvfs_potential'].title()} "
            report += f"(estimated {dvfs_opportunities.get('estimated_power_savings', 0):.1f}% power savings)</p>\n"
            report += "<ul>\n"
            for rec in dvfs_opportunities.get('recommendations', []):
                report += f"<li>{rec}</li>\n"
            report += "</ul>\n"
        else:
            report += f"<p>DVFS potential: {dvfs_opportunities.get('dvfs_potential', 'unknown')}</p>\n"
        
        # Recommendations section
        report += "<h2>Optimization Recommendations</h2>\n"
        if recommendations:
            report += "<ol>\n"
            for rec in recommendations:
                report += f"<li><strong>{rec['description']}</strong> "
                report += f"(Impact: {rec['estimated_impact']}, "
                report += f"Estimated savings: {rec.get('estimated_savings', 0)*100:.1f}%)</li>\n"
            report += "</ol>\n"
        else:
            report += "<p>No specific optimization recommendations generated.</p>\n"
    
    else:  # text format
        report = "GPU ENERGY OPTIMIZATION REPORT\n"
        report += "===========================\n\n"
        
        # Hotspots section
        report += "POWER HOTSPOTS\n"
        report += "-------------\n"
        if hotspots.get('hotspots_found', False):
            report += f"{hotspots['count']} hotspots found, consuming {hotspots['total_energy_percentage']:.1f}% of total energy\n\n"
            for i, period in enumerate(hotspots.get('hotspot_periods', [])):
                report += f"Hotspot {i+1}: at period {period['start_index']}-{period['end_index']}, "
                report += f"avg power: {period['avg_power']:.2f}W, "
                report += f"dominant component: {period['dominant_component']}\n"
        else:
            report += "No significant power hotspots found.\n"
        
        report += "\n"
        
        # Inefficient patterns section
        report += "INEFFICIENT PATTERNS\n"
        report += "-------------------\n"
        if patterns:
            for pattern, details in patterns.items():
                report += f"Pattern: {pattern.replace('_', ' ').title()}\n"
                if 'recommendation' in details:
                    report += f"  Recommendation: {details['recommendation']}\n"
                report += "\n"
        else:
            report += "No significant inefficient patterns detected.\n"
        
        report += "\n"
        
        # DVFS opportunities section
        report += "DVFS OPPORTUNITIES\n"
        report += "-----------------\n"
        report += f"DVFS potential: {dvfs_opportunities.get('dvfs_potential', 'unknown')}\n"
        if dvfs_opportunities.get('dvfs_potential') in ['significant', 'moderate']:
            report += f"Estimated power savings: {dvfs_opportunities.get('estimated_power_savings', 0):.1f}%\n\n"
            for rec in dvfs_opportunities.get('recommendations', []):
                report += f"- {rec}\n"
        
        report += "\n"
        
        # Recommendations section
        report += "OPTIMIZATION RECOMMENDATIONS\n"
        report += "--------------------------\n"
        if recommendations:
            for i, rec in enumerate(recommendations):
                report += f"{i+1}. {rec['description']}\n"
                report += f"   Impact: {rec['estimated_impact']}, "
                report += f"Estimated savings: {rec.get('estimated_savings', 0)*100:.1f}%\n"
                report += "\n"
        else:
            report += "No specific optimization recommendations generated.\n"
    
    return report