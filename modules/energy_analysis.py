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
                    
                    
