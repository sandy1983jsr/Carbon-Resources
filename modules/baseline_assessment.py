"""
Module for baseline assessment of energy, material flows, and process efficiency
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.sankey as sankey

class BaselineAssessment:
    """
    Performs baseline assessment of plant operations
    """
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize the baseline assessment module
        
        Args:
            datasets: Dictionary of datasets loaded from CSV files
        """
        self.datasets = datasets
        self.results = {}
        
    def run_assessment(self):
        """
        Run the complete baseline assessment
        
        Returns:
            Dictionary with assessment results
        """
        self.results['energy_baseline'] = self._analyze_energy_consumption()
        self.results['material_baseline'] = self._analyze_material_flow()
        self.results['kpi_baseline'] = self._calculate_baseline_kpis()
        self.results['correlation_matrix'] = self._analyze_correlations()
        self.results['pca_results'] = self._run_pca()
        
        return self.results
    
    def _analyze_energy_consumption(self):
        """
        Analyze energy consumption patterns
        
        Returns:
            Dictionary with energy analysis results
        """
        results = {}
        
        # Check if energy consumption data is available
        if 'energy_consumption' in self.datasets:
            energy_df = self.datasets['energy_consumption']
            
            # Calculate basic statistics
            results['total_consumption'] = energy_df['kwh_consumed'].sum()
            results['avg_consumption'] = energy_df['kwh_consumed'].mean()
            results['max_consumption'] = energy_df['kwh_consumed'].max()
            
            # Calculate consumption by process area if available
            if 'process_area' in energy_df.columns:
                results['consumption_by_area'] = (
                    energy_df.groupby('process_area')['kwh_consumed'].sum()
                    .sort_values(ascending=False)
                    .to_dict()
                )
            
            # Calculate energy intensity if production data is available
            if 'production' in self.datasets:
                prod_df = self.datasets['production']
                # Merge on date columns assuming both have a date column
                if 'date' in energy_df.columns and 'date' in prod_df.columns:
                    merged = pd.merge(
                        energy_df.groupby('date')['kwh_consumed'].sum().reset_index(),
                        prod_df.groupby('date')['production_tons'].sum().reset_index(),
                        on='date', how='inner'
                    )
                    merged['energy_intensity'] = merged['kwh_consumed'] / merged['production_tons']
                    results['avg_energy_intensity'] = merged['energy_intensity'].mean()
                    results['min_energy_intensity'] = merged['energy_intensity'].min()
                    results['energy_intensity_trend'] = merged[['date', 'energy_intensity']].values.tolist()
        
        return results
    
    def _analyze_material_flow(self):
        """
        Analyze material flow and efficiency
        
        Returns:
            Dictionary with material flow analysis results
        """
        results = {}
        
        # Check if material input and output data is available
        if 'material_input' in self.datasets and 'material_output' in self.datasets:
            input_df = self.datasets['material_input']
            output_df = self.datasets['material_output']
            
            # Calculate total input and output
            results['total_input'] = input_df['quantity_tons'].sum()
            results['total_output'] = output_df['quantity_tons'].sum()
            
            # Calculate material yield
            results['material_yield'] = results['total_output'] / results['total_input']
            
            # Calculate input by material type
            if 'material_type' in input_df.columns:
                results['input_by_type'] = (
                    input_df.groupby('material_type')['quantity_tons'].sum()
                    .sort_values(ascending=False)
                    .to_dict()
                )
            
            # Calculate waste or losses
            results['material_loss'] = results['total_input'] - results['total_output']
            results['material_loss_percentage'] = (results['material_loss'] / results['total_input']) * 100
            
        return results
    
    def _calculate_baseline_kpis(self):
        """
        Calculate key performance indicators for baseline assessment
        
        Returns:
            Dictionary with KPI results
        """
        kpis = {}
        
        # Energy KPIs
        if 'energy_consumption' in self.datasets and 'production' in self.datasets:
            energy_df = self.datasets['energy_consumption']
            prod_df = self.datasets['production']
            
            # Calculate KPIs for the entire dataset period
            total_energy = energy_df['kwh_consumed'].sum()
            total_production = prod_df['production_tons'].sum()
            
            kpis['overall_energy_intensity'] = total_energy / total_production if total_production > 0 else None
            
            # Check if furnace-specific data is available
            if 'furnace_data' in self.datasets:
                furnace_df = self.datasets['furnace_data']
                if 'electrode_paste_consumption_kg' in furnace_df.columns:
                    kpis['avg_electrode_paste_per_ton'] = (
                        furnace_df['electrode_paste_consumption_kg'].sum() / total_production
                    ) if total_production > 0 else None
        
        # Material efficiency KPIs
        if 'material_input' in self.datasets and 'material_output' in self.datasets:
            input_df = self.datasets['material_input']
            output_df = self.datasets['material_output']
            
            total_input = input_df['quantity_tons'].sum()
            total_output = output_df['quantity_tons'].sum()
            
            kpis['material_yield'] = total_output / total_input if total_input > 0 else None
            kpis['material_loss_percentage'] = 100 - (kpis['material_yield'] * 100) if kpis['material_yield'] is not None else None
        
        # Process efficiency KPIs
        if 'process_data' in self.datasets:
            process_df = self.datasets['process_data']
            
            # Calculate equipment utilization if available
            if 'uptime_hours' in process_df.columns and 'total_hours' in process_df.columns:
                kpis['equipment_utilization'] = (
                    process_df['uptime_hours'].sum() / process_df['total_hours'].sum()
                ) if process_df['total_hours'].sum() > 0 else None
            
            # Calculate batch consistency if available
            if 'batch_weight_actual' in process_df.columns and 'batch_weight_target' in process_df.columns:
                deviations = abs(process_df['batch_weight_actual'] - process_df['batch_weight_target'])
                kpis['avg_batch_deviation'] = deviations.mean()
                kpis['batch_deviation_percentage'] = (deviations / process_df['batch_weight_target']).mean() * 100
        
        return kpis
    
    def _analyze_correlations(self):
        """
        Analyze correlations between key variables
        
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_results = {}
        
        # Combine relevant variables from different datasets for correlation analysis
        relevant_vars = {}
        
        # Extract energy variables if available
        if 'energy_consumption' in self.datasets:
            energy_df = self.datasets['energy_consumption']
            if 'kwh_consumed' in energy_df.columns:
                relevant_vars['energy_consumption'] = energy_df['kwh_consumed']
        
        # Extract production variables
        if 'production' in self.datasets:
            prod_df = self.datasets['production']
            if 'production_tons' in prod_df.columns:
                relevant_vars['production_tons'] = prod_df['production_tons']
        
        # Extract material variables
        if 'material_input' in self.datasets:
            input_df = self.datasets['material_input']
            if 'quantity_tons' in input_df.columns:
                relevant_vars['material_input'] = input_df['quantity_tons']
        
        # Extract furnace variables
        if 'furnace_data' in self.datasets:
            furnace_df = self.datasets['furnace_data']
            for col in ['temperature', 'power_factor', 'electrode_paste_consumption_kg']:
                if col in furnace_df.columns:
                    relevant_vars[col] = furnace_df[col]
        
        # Calculate correlation matrix if we have at least 2 variables
        if len(relevant_vars) >= 2:
            # Create a DataFrame from relevant variables
            # Note: This is simplified and would need handling for time alignment in a real system
            corr_df = pd.DataFrame(relevant_vars)
            
            # Calculate correlation matrix
            correlation_matrix = corr_df.corr()
            
            # Store the correlation matrix
            correlation_results['matrix'] = correlation_matrix.to_dict()
            
            # Find strongest correlations (absolute value)
            correlations = []
            for i, row in enumerate(correlation_matrix.index):
                for j, col in enumerate(correlation_matrix.columns):
                    if i < j:  # Upper triangle only to avoid duplicates
                        correlations.append({
                            'var1': row,
                            'var2': col,
                            'correlation': correlation_matrix.loc[row, col]
                        })
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            correlation_results['strongest_correlations'] = correlations[:5]  # Top 5 correlations
        
        return correlation_results
    
    def _run_pca(self):
        """
        Run Principal Component Analysis on process variables
        
        Returns:
            Dictionary with PCA results
        """
        pca_results = {}
        
        # Combine relevant variables for PCA
        process_vars = {}
        
        # Include process variables from different datasets
        datasets_to_check = ['furnace_data', 'process_data', 'energy_consumption']
        
        for dataset_name in datasets_to_check:
            if dataset_name in self.datasets:
                df = self.datasets[dataset_name]
                # Only include numeric columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    process_vars[f"{dataset_name}_{col}"] = df[col]
        
        # Run PCA if we have enough variables
        if len(process_vars) >= 3:
            # Create a DataFrame and drop rows with missing values
            pca_df = pd.DataFrame(process_vars).dropna()
            
            if len(pca_df) >= 10:  # Ensure we have enough samples
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_df)
                
                # Run PCA
                n_components = min(len(pca_df.columns), 5)  # Maximum of 5 components
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # Store results
                pca_results['explained_variance'] = pca.explained_variance_ratio_.tolist()
                pca_results['cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_).tolist()
                pca_results['loadings'] = []
                
                # Calculate and store loadings
                for i, component in enumerate(pca.components_):
                    component_loadings = []
                    for j, var_name in enumerate(pca_df.columns):
                        component_loadings.append({
                            'variable': var_name,
                            'loading': component[j]
                        })
                    # Sort loadings by absolute value
                    component_loadings.sort(key=lambda x: abs(x['loading']), reverse=True)
                    pca_results['loadings'].append({
                        'component': i + 1,
                        'top_variables': component_loadings[:5]  # Top 5 variables for each component
                    })
        
        return pca_results
    
    def generate_reports(self, output_dir: str):
        """
        Generate reports and visualizations for baseline assessment
        
        Args:
            output_dir: Directory to save the reports
        """
        # Create a subdirectory for baseline reports
        baseline_dir = os.path.join(output_dir, 'baseline_assessment')
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Generate energy consumption charts
        if 'energy_baseline' in self.results:
            self._generate_energy_charts(baseline_dir)
        
        # Generate material flow diagrams
        if 'material_baseline' in self.results:
            self._generate_material_flow_diagrams(baseline_dir)
        
        # Generate KPI summary
        if 'kpi_baseline' in self.results:
            self._generate_kpi_summary(baseline_dir)
        
        # Generate correlation heatmap
        if 'correlation_matrix' in self.results:
            self._generate_correlation_heatmap(baseline_dir)
        
        # Generate PCA visualization
        if 'pca_results' in self.results:
            self._generate_pca_visualization(baseline_dir)
            
        # Generate Sankey diagram if we have enough data
        if ('energy_baseline' in self.results and 
            'material_baseline' in self.results):
            self._generate_sankey_diagram(baseline_dir)
        
        # Generate summary report
        self._generate_summary_report(baseline_dir)
    
    def _generate_energy_charts(self, output_dir: str):
        """Generate charts for energy consumption analysis"""
        energy_results = self.results['energy_baseline']
        
        # Energy consumption by area if available
        if 'consumption_by_area' in energy_results:
            plt.figure(figsize=(12, 6))
            areas = list(energy_results['consumption_by_area'].keys())
            values = list(energy_results['consumption_by_area'].values())
            
            # Create a bar chart
            plt.bar(areas, values)
            plt.title('Energy Consumption by Process Area')
            plt.xlabel('Process Area')
            plt.ylabel('Energy Consumption (kWh)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'energy_by_area.png'))
            plt.close()
        
        # Energy intensity trend if available
        if 'energy_intensity_trend' in energy_results:
            trend_data = energy_results['energy_intensity_trend']
            
            plt.figure(figsize=(12, 6))
            dates = [item[0] for item in trend_data]
            intensities = [item[1] for item in trend_data]
            
            plt.plot(dates, intensities, marker='o')
            plt.title('Energy Intensity Trend')
            plt.xlabel('Date')
            plt.ylabel('Energy Intensity (kWh/ton)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'energy_intensity_trend.png'))
            plt.close()
    
    def _generate_material_flow_diagrams(self, output_dir: str):
        """Generate material flow diagrams"""
        material_results = self.results['material_baseline']
        
        # Material input by type if available
        if 'input_by_type' in material_results:
            plt.figure(figsize=(12, 6))
            materials = list(material_results['input_by_type'].keys())
            quantities = list(material_results['input_by_type'].values())
            
            plt.bar(materials, quantities)
            plt.title('Material Input by Type')
            plt.xlabel('Material Type')
            plt.ylabel('Quantity (tons)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'material_input_by_type.png'))
            plt.close()
        
        # Material yield visualization
        plt.figure(figsize=(8, 8))
        plt.pie(
            [material_results['total_output'], material_results['material_loss']],
            labels=['Output', 'Loss/Waste'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#4CAF50', '#F44336']
        )
        plt.title('Material Yield and Loss')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'material_yield_pie.png'))
        plt.close()
    
    def _generate_kpi_summary(self, output_dir: str):
        """Generate KPI summary visualization"""
        kpis = self.results['kpi_baseline']
        
        # Create a summary table for KPIs
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        
        kpi_names = list(kpis.keys())
        kpi_values = [kpis[k] for k in kpi_names]
        
        # Format KPI names for better readability
        formatted_names = [name.replace('_', ' ').title() for name in kpi_names]
        
        # Format values with appropriate units and precision
        formatted_values = []
        for name, value in zip(kpi_names, kpi_values):
            if value is None:
                formatted_values.append('N/A')
            elif 'percentage' in name or 'utilization' in name:
                formatted_values.append(f"{value:.2f}%")
            elif 'energy' in name and 'intensity' in name:
                formatted_values.append(f"{value:.2f} kWh/ton")
            elif 'electrode' in name and 'per_ton' in name:
                formatted_values.append(f"{value:.2f} kg/ton")
            else:
                formatted_values.append(f"{value:.2f}")
        
        # Create a simple table
        table_data = []
        for name, value in zip(formatted_names, formatted_values):
            table_data.append([name, value])
        
        table = plt.table(
            cellText=table_data,
            colLabels=['KPI', 'Value'],
            loc='center',
            cellLoc='center',
            colWidths=[0.7, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Baseline KPI Summary', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kpi_summary.png'))
        plt.close()
    
    def _generate_correlation_heatmap(self, output_dir: str):
        """Generate correlation heatmap"""
        correlation_data = self.results['correlation_matrix']
        
        if 'matrix' in correlation_data:
            corr_matrix = pd.DataFrame(correlation_data['matrix'])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0,
                linewidths=0.5
            )
            plt.title('Correlation Between Key Variables')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
            plt.close()
    
    def _generate_pca_visualization(self, output_dir: str):
        """Generate PCA visualization"""
        pca_data = self.results['pca_results']
        
        if 'explained_variance' in pca_data:
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            variance = pca_data['explained_variance']
            cumulative = pca_data['cumulative_variance']
            
            bars = plt.bar(
                range(1, len(variance) + 1), 
                variance, 
                alpha=0.7, 
                label='Individual Variance'
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2%}',
                    ha='center', 
                    va='bottom',
                    rotation=0
                )
            
            plt.plot(
                range(1, len(cumulative) + 1), 
                cumulative, 
                'ro-', 
                label='Cumulative Variance'
            )
            
            # Add value labels for line plot
            for i, value in enumerate(cumulative):
                plt.text(
                    i + 1 + 0.1, 
                    value - 0.05,
                    f'{value:.2%}',
                    ha='left', 
                    va='center',
                    color='red'
                )
            
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA: Explained Variance by Component')
            plt.xticks(range(1, len(variance) + 1))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
            plt.close()
            
            # Create a visualization of the loadings for the first two components
            if len(pca_data['loadings']) >= 2:
                # This is simplified - a more detailed implementation would be needed
                # for a production system
                pass
    
    def _generate_sankey_diagram(self, output_dir: str):
        """Generate Sankey diagram for energy and material flows"""
        # This is a placeholder - a real implementation would need more detailed data
        # about how energy and materials flow through the system
        pass
    
    def _generate_summary_report(self, output_dir: str):
        """Generate a summary report of all findings"""
        # This would typically create a PDF or HTML report summarizing all findings
        # For simplicity, we'll create a text file summary
        
        with open(os.path.join(output_dir, 'baseline_summary.txt'), 'w') as f:
            f.write("FERRO ALLOY PLANT BASELINE ASSESSMENT\n")
            f.write("====================================\n\n")
            
            # Energy summary
            if 'energy_baseline' in self.results:
                energy = self.results['energy_baseline']
                f.write("ENERGY CONSUMPTION SUMMARY\n")
                f.write("--------------------------\n")
                f.write(f"Total Energy Consumption: {energy.get('total_consumption', 'N/A'):,.2f} kWh\n")
                f.write(f"Average Energy Consumption: {energy.get('avg_consumption', 'N/A'):,.2f} kWh\n")
                if 'avg_energy_intensity' in energy:
                    f.write(f"Average Energy Intensity: {energy['avg_energy_intensity']:,.2f} kWh/ton\n")
                    f.write(f"Best Energy Intensity: {energy['min_energy_intensity']:,.2f} kWh/ton\n")
                f.write("\n")
            
            # Material summary
            if 'material_baseline' in self.results:
                material = self.results['material_baseline']
                f.write("MATERIAL FLOW SUMMARY\n")
                f.write("--------------------\n")
                f.write(f"Total Material Input: {material.get('total_input', 'N/A'):,.2f} tons\n")
                f.write(f"Total Material Output: {material.get('total_output', 'N/A'):,.2f} tons\n")
                f.write(f"Material Yield: {material.get('material_yield', 'N/A'):.2%}\n")
                f.write(f"Material Loss: {material.get('material_loss', 'N/A'):,.2f} tons ")
                f.write(f"({material.get('material_loss_percentage', 'N/A'):.2f}%)\n\n")
            
            # KPI summary
            if 'kpi_baseline' in self.results:
                kpis = self.results['kpi_baseline']
                f.write("KEY PERFORMANCE INDICATORS\n")
                f.write("-------------------------\n")
                for key, value in kpis.items():
                    if value is not None:
                        formatted_key = key.replace('_', ' ').title()
                        if 'percentage' in key or 'utilization' in key:
                            f.write(f"{formatted_key}: {value:.2f}%\n")
                        elif 'energy' in key and 'intensity' in key:
                            f.write(f"{formatted_key}: {value:.2f} kWh/ton\n")
                        else:
                            f.write(f"{formatted_key}: {value:.2f}\n")
                f.write("\n")
            
            # Correlation insights
            if 'correlation_matrix' in self.results and 'strongest_correlations' in self.results['correlation_matrix']:
                correlations = self.results['correlation_matrix']['strongest_correlations']
                f.write("KEY CORRELATIONS\n")
                f.write("---------------\n")
                for corr in correlations:
                    var1 = corr['var1'].replace('_', ' ').title()
                    var2 = corr['var2'].replace('_', ' ').title()
                    value = corr['correlation']
                    f.write(f"{var1} and {var2}: {value:.2f}\n")
                f.write("\n")
            
            # PCA insights
            if 'pca_results' in self.results and 'loadings' in self.results['pca_results']:
                loadings = self.results['pca_results']['loadings']
                f.write("PRINCIPAL COMPONENT ANALYSIS\n")
                f.write("---------------------------\n")
                for component in loadings[:2]:  # First two components only
                    comp_num = component['component']
                    f.write(f"Component {comp_num} - Top Variables:\n")
                    for var in component['top_variables'][:3]:  # Top 3 variables only
                        var_name = var['variable'].replace('_', ' ').title()
                        loading = var['loading']
                        f.write(f"  - {var_name}: {loading:.3f}\n")
                f.write("\n")
            
            # Potential improvement areas
            f.write("POTENTIAL IMPROVEMENT AREAS\n")
            f.write("--------------------------\n")
            
            # Energy improvement potential
            if 'energy_baseline' in self.results:
                energy = self.results['energy_baseline']
                if 'consumption_by_area' in energy:
                    # Identify the largest energy consumers
                    top_areas = sorted(energy['consumption_by_area'].items(), 
                                      key=lambda x: x[1], reverse=True)[:2]
                    f.write("Energy efficiency opportunities:\n")
                    for area, consumption in top_areas:
                        area_name = area.replace('_', ' ').title()
                        f.write(f"  - {area_name}: {consumption:,.2f} kWh ")
                        f.write(f"({consumption/energy['total_consumption']:.1%} of total)\n")
            
            # Material improvement potential
            if 'material_baseline' in self.results:
                material = self.results['material_baseline']
                if material.get('material_loss_percentage', 0) > 5:
                    f.write("Material efficiency opportunities:\n")
                    f.write(f"  - Current material loss: {material['material_loss_percentage']:.1f}%\n")
                    f.write(f"  - Potential savings from 1% improvement: ")
                    f.write(f"{material['total_input'] * 0.01:.2f} tons\n")
            
            f.write("\n")
            f.write("END OF BASELINE ASSESSMENT SUMMARY\n")
