"""
Module for analyzing energy consumption and identifying optimization opportunities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy import stats

class EnergyAnalysis:
    """
    Analyzes energy consumption patterns and identifies optimization opportunities
    """
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize the energy analysis module
        
        Args:
            datasets: Dictionary of datasets loaded from CSV files
        """
        self.datasets = datasets
        self.results = {}
        
    def analyze(self):
        """
        Run the complete energy analysis
        
        Returns:
            Dictionary with analysis results
        """
        # Load energy consumption data
        if 'energy_consumption' not in self.datasets:
            self.results['error'] = "Energy consumption data not available"
            return self.results
            
        energy_df = self.datasets['energy_consumption']
        
        # Analyze energy consumption patterns
        self.results['consumption_patterns'] = self._analyze_consumption_patterns(energy_df)
        
        # Analyze power factor if available
        if 'power_factor' in energy_df.columns:
            self.results['power_factor_analysis'] = self._analyze_power_factor(energy_df)
        
        # Analyze load profiles
        self.results['load_profiles'] = self._analyze_load_profiles(energy_df)
        
        # Analyze energy intensity
        if 'production' in self.datasets:
            self.results['energy_intensity'] = self._analyze_energy_intensity(
                energy_df, self.datasets['production']
            )
        
        # Identify anomalies in energy consumption
        self.results['anomalies'] = self._detect_anomalies(energy_df)
        
        # Perform energy disaggregation if we have detailed data
        if 'process_area' in energy_df.columns:
            self.results['energy_disaggregation'] = self._disaggregate_energy(energy_df)
        
        # Analyze correlations with process parameters
        if 'furnace_data' in self.datasets:
            self.results['energy_process_correlation'] = self._analyze_energy_process_correlation(
                energy_df, self.datasets['furnace_data']
            )
        
        # Calculate energy saving opportunities
        self.results['saving_opportunities'] = self._identify_saving_opportunities()
        
        return self.results
    
    def _analyze_consumption_patterns(self, energy_df: pd.DataFrame):
        """
        Analyze energy consumption patterns
        
        Args:
            energy_df: Energy consumption dataframe
            
        Returns:
            Dictionary with consumption pattern analysis
        """
        results = {}
        
        # Make sure we have datetime information
        if 'timestamp' in energy_df.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(energy_df['timestamp']):
                energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
            
            # Add time components
            energy_df['hour'] = energy_df['timestamp'].dt.hour
            energy_df['day'] = energy_df['timestamp'].dt.day_name()
            energy_df['month'] = energy_df['timestamp'].dt.month_name()
            
            # Hourly patterns
            hourly_consumption = energy_df.groupby('hour')['kwh_consumed'].mean()
            results['hourly_pattern'] = hourly_consumption.to_dict()
            
            # Find peak hours
            peak_hour = hourly_consumption.idxmax()
            peak_consumption = hourly_consumption.max()
            results['peak_hour'] = {
                'hour': int(peak_hour),
                'consumption': float(peak_consumption)
            }
            
            # Daily patterns
            if 'day' in energy_df.columns:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_consumption = energy_df.groupby('day')['kwh_consumed'].mean()
                # Reindex to ensure days are in correct order
                daily_consumption = daily_consumption.reindex(day_order)
                results['daily_pattern'] = daily_consumption.to_dict()
            
            # Monthly patterns
            if 'month' in energy_df.columns:
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly_consumption = energy_df.groupby('month')['kwh_consumed'].mean()
                # Reindex to ensure months are in correct order
                monthly_consumption = monthly_consumption.reindex(month_order)
                results['monthly_pattern'] = monthly_consumption.to_dict()
            
            # Time series decomposition if we have enough data
            if len(energy_df) >= 30:  # Need sufficient data points
                try:
                    # Resample to daily data if we have timestamps
                    daily_data = energy_df.set_index('timestamp').resample('D')['kwh_consumed'].mean()
                    
                    # Fill missing values with forward fill
                    daily_data = daily_data.fillna(method='ffill')
                    
                    if len(daily_data) >= 14:  # Need at least 14 days for decomposition
                        # Perform time series decomposition
                        decomposition = seasonal_decompose(
                            daily_data, 
                            model='additive', 
                            period=7  # Assuming weekly seasonality
                        )
                        
                        # Extract components
                        results['time_series'] = {
                            'trend': decomposition.trend.dropna().tolist(),
                            'seasonal': decomposition.seasonal.dropna().tolist(),
                            'residual': decomposition.resid.dropna().tolist()
                        }
                except Exception as e:
                    results['time_series_error'] = str(e)
        
        # Overall consumption statistics
        results['total_consumption'] = float(energy_df['kwh_consumed'].sum())
        results['average_consumption'] = float(energy_df['kwh_consumed'].mean())
        results['std_consumption'] = float(energy_df['kwh_consumed'].std())
        results['min_consumption'] = float(energy_df['kwh_consumed'].min())
        results['max_consumption'] = float(energy_df['kwh_consumed'].max())
        
        return results
    
    def _analyze_power_factor(self, energy_df: pd.DataFrame):
        """
        Analyze power factor data
        
        Args:
            energy_df: Energy consumption dataframe
            
        Returns:
            Dictionary with power factor analysis
        """
        results = {}
        
        if 'power_factor' in energy_df.columns:
            power_factor = energy_df['power_factor']
            
            # Basic statistics
            results['average_pf'] = float(power_factor.mean())
            results['min_pf'] = float(power_factor.min())
            results['max_pf'] = float(power_factor.max())
            
            # Calculate percentage of time below optimal PF (usually 0.95)
            below_optimal = (power_factor < 0.95).mean() * 100
            results['below_optimal_percentage'] = float(below_optimal)
            
            # Calculate potential savings from improving power factor
            if results['average_pf'] < 0.95:
                # Simplified calculation - actual savings would depend on tariff structure
                potential_reduction = 1 - (results['average_pf'] / 0.95)
                estimated_savings = potential_reduction * energy_df['kwh_consumed'].sum() * 0.05  # Assuming 5% of energy cost
                results['potential_savings'] = float(estimated_savings)
                results['improved_pf_target'] = 0.95
            
            # Check if there are specific areas or equipment with poor power factor
            if 'process_area' in energy_df.columns:
                pf_by_area = energy_df.groupby('process_area')['power_factor'].mean().sort_values()
                areas_below_optimal = pf_by_area[pf_by_area < 0.95].to_dict()
                results['areas_below_optimal'] = areas_below_optimal
        
        return results
    
    def _analyze_load_profiles(self, energy_df: pd.DataFrame):
        """
        Analyze load profiles to identify patterns and optimization opportunities
        
        Args:
            energy_df: Energy consumption dataframe
            
        Returns:
            Dictionary with load profile analysis
        """
        results = {}
        
        # Check if we have timestamp data
        if 'timestamp' in energy_df.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(energy_df['timestamp']):
                energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
            
            # Calculate load factor (average load / peak load)
            if 'kwh_consumed' in energy_df.columns:
                # Group by day to calculate daily load factors
                daily_energy = energy_df.groupby(energy_df['timestamp'].dt.date)['kwh_consumed']
                daily_avg = daily_energy.mean()
                daily_max = daily_energy.max()
                
                daily_load_factors = (daily_avg / daily_max).dropna()
                
                results['average_load_factor'] = float(daily_load_factors.mean())
                results['min_load_factor'] = float(daily_load_factors.min())
                results['max_load_factor'] = float(daily_load_factors.max())
                
                # Calculate overall load factor
                overall_avg = energy_df['kwh_consumed'].mean()
                overall_max = energy_df['kwh_consumed'].max()
                results['overall_load_factor'] = float(overall_avg / overall_max)
                
                # Identify days with poor load factors
                poor_load_factor_days = daily_load_factors[daily_load_factors < 0.6].sort_values()
                results['poor_load_factor_days'] = {str(date): float(factor) for date, factor in poor_load_factor_days.items()}
                
                # Attempt to cluster load profiles to identify patterns
                try:
                    # Create hourly profiles for each day
                    energy_df['date'] = energy_df['timestamp'].dt.date
                    energy_df['hour'] = energy_df['timestamp'].dt.hour
                    
                    # Pivot to create daily load profiles
                    daily_profiles = energy_df.pivot_table(
                        index='date', 
                        columns='hour', 
                        values='kwh_consumed',
                        aggfunc='mean'
                    ).fillna(method='ffill')
                    
                    if len(daily_profiles) >= 5:  # Need enough days for meaningful clustering
                        # Normalize profiles for clustering
                        scaler = StandardScaler()
                        normalized_profiles = scaler.fit_transform(daily_profiles)
                        
                        # Determine optimal number of clusters (simple approach)
                        max_clusters = min(5, len(daily_profiles) // 2)
                        inertias = []
                        for k in range(1, max_clusters + 1):
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            kmeans.fit(normalized_profiles)
                            inertias.append(kmeans.inertia_)
                        
                        # Simple elbow method - find point where inertia decrease slows
                        k_selected = 2  # Default to 2 clusters
                        if len(inertias) > 2:
                            # Calculate rate of decrease
                            decreases = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
                            # Find where decrease slows down significantly
                            for i in range(len(decreases)-1):
                                if decreases[i] > decreases[i+1] * 2:  # If decrease rate halves
                                    k_selected = i + 2  # +2 because we start from k=1 and i is 0-indexed
                                    break
                        
                        # Cluster with selected k
                        kmeans = KMeans(n_clusters=k_selected, random_state=42)
                        clusters = kmeans.fit_predict(normalized_profiles)
                        
                        # Extract cluster centers and denormalize
                        centers = kmeans.cluster_centers_
                        denormalized_centers = scaler.inverse_transform(centers)
                        
                        # Store cluster profiles
                        cluster_profiles = {}
                        for i in range(k_selected):
                            cluster_profiles[f'cluster_{i+1}'] = {
                                'profile': denormalized_centers[i].tolist(),
                                'count': int((clusters == i).sum()),
                                'percentage': float((clusters == i).mean() * 100)
                            }
                        
                        results['load_clusters'] = cluster_profiles
                        
                        # Identify the most efficient cluster (lowest average consumption)
                        avg_consumptions = [sum(profile['profile']) for profile in cluster_profiles.values()]
                        most_efficient_idx = np.argmin(avg_consumptions)
                        results['most_efficient_cluster'] = f'cluster_{most_efficient_idx+1}'
                        
                        # Calculate potential savings if all days matched most efficient cluster
                        current_avg = energy_df['kwh_consumed'].mean()
                        efficient_avg = sum(cluster_profiles[results['most_efficient_cluster']]['profile']) / 24
                        if efficient_avg < current_avg:
                            potential_savings = (current_avg - efficient_avg) / current_avg * 100
                            results['potential_load_savings_percentage'] = float(potential_savings)
                
                except Exception as e:
                    results['clustering_error'] = str(e)
        
        return results
    
    def _analyze_energy_intensity(self, energy_df: pd.DataFrame, production_df: pd.DataFrame):
        """
        Analyze energy intensity (energy per unit of production)
        
        Args:
            energy_df: Energy consumption dataframe
            production_df: Production data dataframe
            
        Returns:
            Dictionary with energy intensity analysis
        """
        results = {}
        
        # Check if we have the necessary columns
        if 'kwh_consumed' in energy_df.columns and 'production_tons' in production_df.columns:
            # Check if we can align the data by timestamp
            if 'timestamp' in energy_df.columns and 'timestamp' in production_df.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(energy_df['timestamp']):
                    energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
                if not pd.api.types.is_datetime64_any_dtype(production_df['timestamp']):
                    production_df['timestamp'] = pd.to_datetime(production_df['timestamp'])
                
                # Align to daily data for matching
                energy_df['date'] = energy_df['timestamp'].dt.date
                production_df['date'] = production_df['timestamp'].dt.date
                
                daily_energy = energy_df.groupby('date')['kwh_consumed'].sum().reset_index()
                daily_production = production_df.groupby('date')['production_tons'].sum().reset_index()
                
                # Merge the daily data
                merged = pd.merge(daily_energy, daily_production, on='date', how='inner')
                
                if len(merged) > 0:
                    # Calculate energy intensity
                    merged['energy_intensity'] = merged['kwh_consumed'] / merged['production_tons']
                    
                    # Calculate statistics
                    results['average_intensity'] = float(merged['energy_intensity'].mean())
                    results['min_intensity'] = float(merged['energy_intensity'].min())
                    results['max_intensity'] = float(merged['energy_intensity'].max())
                    results['std_intensity'] = float(merged['energy_intensity'].std())
                    
                    # Find best performing days
                    best_days = merged.nsmallest(3, 'energy_intensity')
                    results['best_days'] = [
                        {
                            'date': str(row['date']),
                            'energy_intensity': float(row['energy_intensity']),
                            'production_tons': float(row['production_tons'])
                        }
                        for _, row in best_days.iterrows()
                    ]
                    
                    # Find worst performing days
                    worst_days = merged.nlargest(3, 'energy_intensity')
                    results['worst_days'] = [
                        {
                            'date': str(row['date']),
                            'energy_intensity': float(row['energy_intensity']),
                            'production_tons': float(row['production_tons'])
                        }
                        for _, row in worst_days.iterrows()
                    ]
                    
                    # Calculate potential savings if all days matched best performance
                    if results['min_intensity'] < results['average_intensity']:
                        improvement_potential = (results['average_intensity'] - results['min_intensity']) / results['average_intensity']
                        total_energy = energy_df['kwh_consumed'].sum()
                        potential_savings = improvement_potential * total_energy
                        results['potential_savings_kwh'] = float(potential_savings)
                        results['improvement_percentage'] = float(improvement_potential * 100)
                    
                    # Check if there's a correlation between production volume and energy intensity
                    correlation = merged['production_tons'].corr(merged['energy_intensity'])
                    results['production_intensity_correlation'] = float(correlation)
                    
                    # Check if higher production volumes lead to better efficiency
                    if correlation < -0.3:  # Moderate negative correlation
                        results['scale_efficiency'] = True
                        results['interpretation'] = "Higher production volumes correlate with better energy efficiency"
                    elif correlation > 0.3:  # Moderate positive correlation
                        results['scale_efficiency'] = False
                        results['interpretation'] = "Higher production volumes correlate with worse energy efficiency"
                    else:
                        results['scale_efficiency'] = None
                        results['interpretation'] = "No strong correlation between production volume and energy efficiency"
            else:
                # If we can't match by timestamp, use overall averages
                total_energy = energy_df['kwh_consumed'].sum()
                total_production = production_df['production_tons'].sum()
                
                if total_production > 0:
                    overall_intensity = total_energy / total_production
                    results['overall_energy_intensity'] = float(overall_intensity)
        
        return results
    
    def _detect_anomalies(self, energy_df: pd.DataFrame):
        """
        Detect anomalies in energy consumption
        
        Args:
            energy_df: Energy consumption dataframe
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {}
        
        if 'kwh_consumed' in energy_df.columns:
            # Z-score method for anomaly detection
            mean = energy_df['kwh_consumed'].mean()
            std = energy_df['kwh_consumed'].std()
            
            if std > 0:  # Avoid division by zero
                energy_df['z_score'] = (energy_df['kwh_consumed'] - mean) / std
                
                # Detect high anomalies (z-score > 3)
                high_anomalies = energy_df[energy_df['z_score'] > 3]
                
                # Detect low anomalies (z-score < -3)
                low_anomalies = energy_df[energy_df['z_score'] < -3]
                
                results['high_anomalies_count'] = len(high_anomalies)
                results['low_anomalies_count'] = len(low_anomalies)
                
                # Extract top anomalies for reporting
                if 'timestamp' in energy_df.columns:
                    high_anomalies_list = []
                    for _, row in high_anomalies.nlargest(5, 'z_score').iterrows():
                        anomaly = {
                            'timestamp': str(row['timestamp']),
                            'consumption': float(row['kwh_consumed']),
                            'z_score': float(row['z_score']),
                            'deviation_percentage': float((row['kwh_consumed'] - mean) / mean * 100)
                        }
                        high_anomalies_list.append(anomaly)
                    
                    low_anomalies_list = []
                    for _, row in low_anomalies.nsmallest(5, 'z_score').iterrows():
                        anomaly = {
                            'timestamp': str(row['timestamp']),
                            'consumption': float(row['kwh_consumed']),
                            'z_score': float(row['z_score']),
                            'deviation_percentage': float((row['kwh_consumed'] - mean) / mean * 100)
                        }
                        low_anomalies_list.append(anomaly)
                    
                    results['top_high_anomalies'] = high_anomalies_list
                    results['top_low_anomalies'] = low_anomalies_list
                
                # Calculate impact of anomalies
                if len(high_anomalies) > 0:
                    excess_energy = high_anomalies['kwh_consumed'].sum() - (mean * len(high_anomalies))
                    results['excess_energy_from_high_anomalies'] = float(excess_energy)
                    results['high_anomalies_impact_percentage'] = float(excess_energy / energy_df['kwh_consumed'].sum() * 100)
            
            # Try more advanced anomaly detection if we have timestamp data
            if 'timestamp' in energy_df.columns and len(energy_df) >= 30:
                try:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(energy_df['timestamp']):
                        energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
                    
                    # Resample to hourly data
                    hourly_data = energy_df.set_index('timestamp').resample('H')['kwh_consumed'].mean()
                    
                    # Fill missing values
                    hourly_data = hourly_data.fillna(method='ffill')
                    
                    # Add time-based features
                    hourly_index = pd.DataFrame(index=hourly_data.index)
                    hourly_index['hour'] = hourly_index.index.hour
                    hourly_index['dayofweek'] = hourly_index.index
