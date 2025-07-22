"""
Module for loading and preprocessing data from CSV files
"""
import os
import pandas as pd
import yaml
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and preprocessing of all data files for the optimization system
    """
    
    def __init__(self, data_dir: str, config_path: str):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing input CSV files
            config_path: Path to configuration YAML file
        """
        self.data_dir = data_dir
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        # Expected data files and their schemas
        self.expected_files = self.config.get('data_files', {})
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets defined in the configuration
        
        Returns:
            Dictionary mapping dataset names to pandas DataFrames
        """
        datasets = {}
        
        for dataset_name, file_info in self.expected_files.items():
            file_path = os.path.join(self.data_dir, file_info['filename'])
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                # Load data with appropriate parameters
                df = pd.read_csv(
                    file_path,
                    parse_dates=file_info.get('date_columns', []),
                    index_col=file_info.get('index_col', None)
                )
                
                # Apply preprocessing steps
                df = self._preprocess_dataframe(df, file_info.get('preprocessing', {}))
                
                datasets[dataset_name] = df
                logger.info(f"Successfully loaded {dataset_name} with shape {df.shape}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        # Create derived datasets if needed
        for derived_name, derived_info in self.config.get('derived_datasets', {}).items():
            try:
                source_df = datasets[derived_info['source']]
                # Apply transformations based on configuration
                derived_df = self._derive_dataset(source_df, derived_info)
                datasets[derived_name] = derived_df
            except Exception as e:
                logger.error(f"Error creating derived dataset {derived_name}: {str(e)}")
        
        return datasets
    
    def _preprocess_dataframe(self, df: pd.DataFrame, preprocessing_config: dict) -> pd.DataFrame:
        """
        Apply preprocessing steps to a dataframe based on configuration
        
        Args:
            df: Input dataframe
            preprocessing_config: Configuration dictionary with preprocessing steps
            
        Returns:
            Preprocessed dataframe
        """
        # Handle missing values
        if 'handle_missing' in preprocessing_config:
            missing_strategy = preprocessing_config['handle_missing']
            if missing_strategy == 'drop':
                df = df.dropna()
            elif missing_strategy == 'fill_mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'fill_median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'fill_zero':
                df = df.fillna(0)
            elif isinstance(missing_strategy, dict):
                for col, value in missing_strategy.items():
                    if col in df.columns:
                        df[col] = df[col].fillna(value)
        
        # Remove outliers if configured
        if 'remove_outliers' in preprocessing_config:
            outlier_config = preprocessing_config['remove_outliers']
            for col, threshold in outlier_config.items():
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    mean, std = df[col].mean(), df[col].std()
                    df = df[(df[col] > mean - threshold * std) & 
                            (df[col] < mean + threshold * std)]
        
        # Add derived columns if needed
        if 'add_columns' in preprocessing_config:
            for new_col, formula in preprocessing_config['add_columns'].items():
                if 'ratio' in formula:
                    col1, col2 = formula['ratio']
                    if col1 in df.columns and col2 in df.columns:
                        df[new_col] = df[col1] / df[col2].replace(0, np.nan)
                elif 'sum' in formula:
                    cols = formula['sum']
                    df[new_col] = df[cols].sum(axis=1)
        
        return df
    
    def _derive_dataset(self, source_df: pd.DataFrame, derived_info: dict) -> pd.DataFrame:
        """
        Create a derived dataset through transformations on a source dataset
        
        Args:
            source_df: Source dataframe
            derived_info: Configuration for the derivation
            
        Returns:
            Derived dataframe
        """
        # Aggregation derivation
        if derived_info.get('type') == 'aggregate':
            groupby_cols = derived_info.get('groupby', [])
            agg_dict = derived_info.get('aggregations', {})
            return source_df.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        # Filter derivation
        elif derived_info.get('type') == 'filter':
            filter_cond = derived_info.get('condition', {})
            filtered_df = source_df
            for col, value in filter_cond.items():
                if col in source_df.columns:
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[col].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[col] == value]
            return filtered_df
        
        # Time-based derivation
        elif derived_info.get('type') == 'time_transform':
            if 'timestamp' in source_df.columns:
                df = source_df.copy()
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day'] = pd.to_datetime(df['timestamp']).dt.day_name()
                df['month'] = pd.to_datetime(df['timestamp']).dt.month
                return df
        
        # Default: return copy of source
        return source_df.copy()
