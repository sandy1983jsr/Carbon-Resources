"""
Module for running what-if scenarios to evaluate potential process optimizations
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import os
import matplotlib.pyplot as plt
import seaborn as sns

class WhatIfEngine:
    """
    Runs what-if scenarios to evaluate potential process optimizations
    """
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize the what-if analysis engine
        
        Args:
            datasets: Dictionary of datasets loaded from CSV files
        """
        self.datasets = datasets
        self.results = {}
        self.scenarios = []
        
    def generate_scenarios(self):
        """
        Generate potential optimization scenarios
        
        Returns:
            List of scenarios
        """
        scenarios = []
        
        # Generate energy optimization scenarios
        energy_scenarios = self._generate_energy_scenarios()
        scenarios.extend(energy_scenarios)
        
        # Generate material optimization scenarios
        material_scenarios = self._generate_material_scenarios()
        scenarios.extend(material_scenarios)
        
        # Generate furnace optimization scenarios
        furnace_scenarios = self._generate_furnace_scenarios()
        scenarios.extend(furnace_scenarios)
        
        # Generate electrode optimization scenarios
        electrode_scenarios = self._generate_electrode_scenarios()
        scenarios.extend(electrode_scenarios)
        
        # Store scenarios for later use
        self.scenarios = scenarios
        return scenarios
    
    def _generate_energy_scenarios(self):
        """
        Generate energy-related optimization scenarios
        
        Returns:
            List of energy optimization scenarios
        """
        scenarios = []
        
        # Check if we have energy consumption data
        if 'energy_consumption' in self.datasets:
            energy_df = self.datasets['energy_consumption']
            
            # Scenario 1: Power factor improvement
            if 'power_factor' in energy_df.columns:
                avg_pf = energy_df['power_factor'].mean()
                if avg_pf < 0.95:
                    # Simplified calculation - actual savings would depend on tariff structure
                    potential_reduction = 1 - (avg_pf / 0.95)
                    energy_savings = potential_reduction * energy_df['kwh_consumed'].sum() * 0.05
                    
                    scenarios.append({
                        'id': 'pf_improvement',
                        'name': 'Power Factor Improvement',
                        'description': f'Improve power factor from current {avg_pf:.2f} to 0.95',
                        'category': 'energy',
                        'implementation_difficulty': 'medium',
                        'energy_savings_kwh': float(energy_savings),
                        'energy_savings_percentage': float(potential_reduction * 5),  # 5% of total energy cost
                        'material_savings_tons': 0,
                        'cost_savings_percentage': float(potential_reduction * 5),
                        'investment_level': 'low',
                        'payback_period_months': 12,
                        'actions': [
                            'Install power factor correction capacitors',
                            'Replace old motor drives with VFDs',
                            'Implement reactive power management system'
                        ]
                    })
            
            # Scenario 2: Load balancing
            if 'timestamp' in energy_df.columns:
                # Check if we have hourly patterns
                energy_df['hour'] = pd.to_datetime(energy_df['timestamp']).dt.hour
                hourly_avg = energy_df.groupby('hour')['kwh_consumed'].mean()
                
                peak_hour = hourly_avg.idxmax()
                peak_consumption = hourly_avg.max()
                avg_consumption = hourly_avg.mean()
                
                if peak_consumption > avg_consumption * 1.5:  # Significant peak
                    potential_savings = (peak_consumption - avg_consumption * 1.2) * 0.5  # Assume 50% of excess peak can be shifted
                    total_days = energy_df['timestamp'].dt.date.nunique()
                    annual_savings = potential_savings * total_days / len(energy_df['timestamp'].dt.date.unique()) * 365
                    
                    scenarios.append({
                        'id': 'load_balancing',
                        'name': 'Load Balancing',
                        'description': f'Shift loads from peak hour ({peak_hour}) to off-peak hours',
                        'category': 'energy',
                        'implementation_difficulty': 'medium',
                        'energy_savings_kwh': float(annual_savings),
                        'energy_savings_percentage': float((annual_savings / energy_df['kwh_consumed'].sum()) * 100),
                        'material_savings_tons': 0,
                        'cost_savings_percentage': float((annual_savings / energy_df['kwh_consumed'].sum()) * 100),
                        'investment_level': 'low',
                        'payback_period_months': 6,
                        'actions': [
                            'Implement load scheduling system',
                            'Shift non-critical processes to off-peak hours',
                            'Install energy storage for peak shaving'
                        ]
                    })
        
        # Scenario 3: General energy efficiency improvements
        # This is a more generic scenario that can be applied even without detailed data
        scenarios.append({
            'id': 'energy_efficiency',
            'name': 'General Energy Efficiency Improvements',
            'description': 'Implement standard energy efficiency measures across the plant',
            'category': 'energy',
            'implementation_difficulty': 'medium',
            'energy_savings_kwh': 500000,  # Example value
            'energy_savings_percentage': 7.5,
            'material_savings_tons': 0,
            'cost_savings_percentage': 7.5,
            'investment_level': 'medium',
            'payback_period_months': 18,
            'actions': [
                'Replace inefficient lighting with LED',
                'Improve insulation on furnaces and equipment',
                'Implement energy management system',
                'Regular maintenance of motors and drives',
                'Heat recovery systems'
            ]
        })
        
        return scenarios
    
    def _generate_material_scenarios(self):
        """
        Generate material-related optimization scenarios
        
        Returns:
            List of material optimization scenarios
        """
        scenarios = []
        
        # Check if we have material input and output data
        if 'material_input' in self.datasets and 'material_output' in self.datasets:
            input_df = self.datasets['material_input']
            output_df = self.datasets['material_output']
            
            # Calculate total input and output
            total_input = input_df['quantity_tons'].sum()
            total_output = output_df['quantity_tons'].sum()
            
            # Calculate material yield
            material_yield = total_output / total_input if total_input > 0 else 0
            material_loss = total_input - total_output
            
            # Scenario 1: Reduce material loss in conveyors
            if material_yield < 0.95:  # If yield is less than 95%
                conveyor_loss_reduction = material_loss * 0.2  # Assume 20% of losses occur in conveyors and can be addressed
                
                scenarios.append({
                    'id': 'conveyor_loss_reduction',
                    'name': 'Conveyor Material Loss Reduction',
                    'description': 'Reduce material losses during transport in conveyors',
                    'category': 'material',
                    'implementation_difficulty': 'low',
                    'energy_savings_kwh': 0,
                    'energy_savings_percentage': 0,
                    'material_savings_tons': float(conveyor_loss_reduction),
                    'material_savings_percentage': float(conveyor_loss_reduction / total_input * 100),
                    'cost_savings_percentage': float(conveyor_loss_reduction / total_input * 100 * 0.8),  # Material costs ~80% of total
                    'investment_level': 'low',
                    'payback_period_months': 6,
                    'actions': [
                        'Improve sealing on transfer points',
                        'Install material catch systems',
                        'Regular maintenance of belt alignment',
                        'Install weighing systems for real-time loss detection'
                    ]
                })
            
            # Scenario 2: Improve batch weighing accuracy
            batch_weighing_improvement = material_loss * 0.15  # Assume 15% of losses due to inaccurate weighing
            
            scenarios.append({
                'id': 'batch_weighing_accuracy',
                'name': 'Batch Weighing Accuracy Improvement',
                'description': 'Improve accuracy of batch weighing to reduce over-dosing',
                'category': 'material',
                'implementation_difficulty': 'medium',
                'energy_savings_kwh': 0,
                'energy_savings_percentage': 0,
                'material_savings_tons': float(batch_weighing_improvement),
                'material_savings_percentage': float(batch_weighing_improvement / total_input * 100),
                'cost_savings_percentage': float(batch_weighing_improvement / total_input * 100 * 0.8),
                'investment_level': 'medium',
                'payback_period_months': 12,
                'actions': [
                    'Upgrade weighing system controllers',
                    'Implement calibration program',
                    'Install digital monitoring system',
                    'Implement statistical process control for batching'
                ]
            })
        
        # Scenario 3: Improve metal-slag separation
        scenarios.append({
            'id': 'metal_slag_separation',
            'name': 'Improved Metal-Slag Separation',
            'description': 'Enhance metal recovery from slag',
            'category': 'material',
            'implementation_difficulty': 'high',
            'energy_savings_kwh': 0,
            'energy_savings_percentage': 0,
            'material_savings_tons': 120,  # Example value
            'material_savings_percentage': 2.5,
            'cost_savings_percentage': 2.0,
            'investment_level': 'high',
            'payback_period_months': 24,
            'actions': [
                'Install advanced separation technology',
                'Optimize slag chemistry',
                'Implement slag cooling control',
                'Process slag for secondary recovery'
            ]
        })
        
        return scenarios
    
    def _generate_furnace_scenarios(self):
        """
        Generate furnace-related optimization scenarios
        
        Returns:
            List of furnace optimization scenarios
        """
        scenarios = []
        
        # Check if we have furnace data
        if 'furnace_data' in self.datasets:
            furnace_df = self.datasets['furnace_data']
            
            # Scenario 1: Optimize furnace temperature control
            if 'temperature' in furnace_df.columns:
                temperature_std = furnace_df['temperature'].std()
                
                if temperature_std > 10:  # High temperature variability
                    energy_savings = 0.03  # Assume 3% energy savings from better temperature control
                    
                    if 'kwh_consumed' in furnace_df.columns:
                        energy_savings_kwh = furnace_df['kwh_consumed'].sum() * energy_savings
                    else:
                        energy_savings_kwh = 200000  # Default value if actual consumption not available
                    
                    scenarios.append({
                        'id': 'temperature_control',
                        'name': 'Furnace Temperature Control Optimization',
                        'description': 'Implement advanced control systems to stabilize furnace temperature',
                        'category': 'furnace',
                        'implementation_difficulty': 'medium',
                        'energy_savings_kwh': float(energy_savings_kwh),
                        'energy_savings_percentage': float(energy_savings * 100),
                        'material_savings_tons': 0,
                        'cost_savings_percentage': float(energy_savings * 100 * 0.7),  # Energy is ~70% of furnace operating cost
                        'investment_level': 'medium',
                        'payback_period_months': 14,
                        'actions': [
                            'Install advanced temperature sensors',
                            'Implement PID control algorithms',
                            'Thermal imaging for hot spot detection',
                            'Operator training on optimal temperature management'
                        ]
                    })
            
            # Scenario 2: Furnace lining optimization
            scenarios.append({
                'id': 'furnace_lining',
                'name': 'Furnace Lining Optimization',
                'description': 'Improve furnace lining to reduce heat loss and extend campaign life',
                'category': 'furnace',
                'implementation_difficulty': 'high',
                'energy_savings_kwh': 150000,  # Example value
                'energy_savings_percentage': 2.5,
                'material_savings_tons': 0,
                'cost_savings_percentage': 3.0,  # Includes maintenance savings
                'investment_level': 'high',
                'payback_period_months': 18,
                'actions': [
                    'Replace with advanced refractory materials',
                    'Implement improved installation practices',
                    'Regular thermal scanning for early detection of failures',
                    'Optimize cooling systems'
                ]
            })
        
        # Scenario 3: Raw material mix optimization
        scenarios.append({
            'id': 'raw_material_mix',
            'name': 'Raw Material Mix Optimization',
            'description': 'Optimize the composition of raw materials to improve furnace efficiency',
            'category': 'furnace',
            'implementation_difficulty': 'medium',
            'energy_savings_kwh': 250000,  # Example value
            'energy_savings_percentage': 4.0,
            'material_savings_tons': 150,
            'material_savings_percentage': 3.0,
            'cost_savings_percentage': 5.0,
            'investment_level': 'low',
            'payback_period_months': 6,
            'actions': [
                'Implement raw material quality testing',
                'Develop optimal mix models based on material properties',
                'Install inline analyzers for real-time adjustments',
                'Train operators on optimal charging practices'
            ]
        })
        
        return scenarios
    
    def _generate_electrode_scenarios(self):
        """
        Generate electrode-related optimization scenarios
        
        Returns:
            List of electrode optimization scenarios
        """
        scenarios = []
        
        # Check if we have electrode paste consumption data
        electrode_paste_consumption = None
        if 'furnace_data' in self.datasets and 'electrode_paste_consumption_kg' in self.datasets['furnace_data'].columns:
            electrode_paste_consumption = self.datasets['furnace_data']['electrode_paste_consumption_kg'].sum()
        
        # Scenario 1: Electrode paste quality improvement
        if electrode_paste_consumption is not None:
            paste_savings = electrode_paste_consumption * 0.10  # Assume 10% savings from quality improvement
            
            scenarios.append({
                'id': 'electrode_paste_quality',
                'name': 'Electrode Paste Quality Improvement',
                'description': 'Use higher quality electrode paste to reduce consumption',
                'category': 'electrode',
                'implementation_difficulty': 'medium',
                'energy_savings_kwh': float(paste_savings * 0.5),  # Each kg saved also saves some energy
                'energy_savings_percentage': 1.0,
                'material_savings_tons': float(paste_savings / 1000),  # Convert kg to tons
                'material_savings_percentage': 10.0,
                'cost_savings_percentage': 2.5,
                'investment_level': 'medium',
                'payback_period_months': 12,
                'actions': [
                    'Source higher quality electrode paste',
                    'Implement quality testing program',
                    'Optimize paste storage conditions',
                    'Train staff on proper handling procedures'
                ]
            })
        
        # Scenario 2: Electrode positioning control
        scenarios.append({
            'id': 'electrode_positioning',
            'name': 'Electrode Positioning Control',
            'description': 'Implement precise control of electrode positioning',
            'category': 'electrode',
            'implementation_difficulty': 'medium',
            'energy_savings_kwh': 180000,  # Example value
            'energy_savings_percentage': 3.0,
            'material_savings_tons': 5,  # Small amount of electrode material saved
            'material_savings_percentage': 5.0,  # Of electrode material
            'cost_savings_percentage': 2.5,
            'investment_level': 'medium',
            'payback_period_months': 15,
            'actions': [
                'Install advanced positioning systems',
                'Implement real-time monitoring',
                'Develop automated adjustment algorithms',
                'Train operators on optimal electrode management'
            ]
        })
        
        # Scenario 3: Electrode cooling optimization
        scenarios.append({
            'id': 'electrode_cooling',
            'name': 'Electrode Cooling Optimization',
            'description': 'Optimize cooling to extend electrode life',
            'category': 'electrode',
            'implementation_difficulty': 'low',
            'energy_savings_kwh': 50000,  # Example value
            'energy_savings_percentage': 0.8,
            'material_savings_tons': 8,
            'material_savings_percentage': 8.0,  # Of electrode material
            'cost_savings_percentage': 1.5,
            'investment_level': 'low',
            'payback_period_months': 9,
            'actions': [
                'Optimize water cooling circuits',
                'Implement temperature monitoring',
                'Adjust cooling based on operating conditions',
                'Preventive maintenance program for cooling systems'
            ]
        })
        
        return scenarios
    
    def run_scenarios(self, scenarios: List[dict]):
        """
        Run what-if simulations for the provided scenarios
        
        Args:
            scenarios: List of scenarios to simulate
        """
        scenario_results = {}
        
        for scenario in scenarios:
            # Calculate ROI and other financial metrics
            investment_levels = {
                'low': 50000,
                'medium': 150000,
                'high': 350000
            }
            
            # Estimate investment cost based on investment level
            investment_cost = investment_levels.get(scenario['investment_level'], 100000)
            
            # Calculate annual savings
            if 'energy_consumption' in self.datasets:
                energy_cost_per_kwh = 0.08  # Assumed electricity cost in $/kWh
                energy_savings_annual = scenario.get('energy_savings_kwh', 0) * energy_cost_per_kwh
            else:
                energy_savings_annual = 0
                
            if 'material_input' in self.datasets:
                # Estimate material cost based on input data
                avg_material_cost_per_ton = 800  # Assumed average material cost in $/ton
                material_savings_annual = scenario.get('material_savings_tons', 0) * avg_material_cost_per_ton
            else:
                material_savings_annual = 0
                
            # Calculate total annual savings
            total_annual_savings = energy_savings_annual + material_savings_annual
            
            # Calculate ROI, payback period, and NPV
            if total_annual_savings > 0:
                roi = (total_annual_savings / investment_cost) * 100
                payback_months = (investment_cost / total_annual_savings) * 12
                
                # Simple NPV calculation (5-year horizon, 10% discount rate)
                discount_rate = 0.10
                npv = -investment_cost
                for year in range(1, 6):
                    npv += total_annual_savings / ((1 + discount_rate) ** year)
            else:
                roi = 0
                payback_months = float('inf')
                npv = -investment_cost
            
            # Store results for this scenario
            scenario_results[scenario['id']] = {
                'name': scenario['name'],
                'investment_cost': float(investment_cost),
                'annual_energy_savings': float(energy_savings_annual),
                'annual_material_savings': float(material_savings_annual),
                'total_annual_savings': float(total_annual_savings),
                'roi_percentage': float(roi),
                'payback_months': float(payback_months),
                'npv_5year': float(npv),
                'implementation_difficulty': scenario['implementation_difficulty']
            }
        
        # Calculate combined scenarios
        combined_scenarios = self._generate_combined_scenarios(scenarios, scenario_results)
        
        # Store all results
        self.results = {
            'individual_scenarios': scenario_results,
            'combined_scenarios': combined_scenarios
        }
    
    def _generate_combined_scenarios(self, scenarios: List[dict], scenario_results: Dict[str, dict]):
        """
        Generate combined scenarios from individual ones
        
        Args:
            scenarios: List of individual scenarios
            scenario_results: Results from individual scenarios
            
        Returns:
            Dictionary with combined scenarios
        """
        combined_scenarios = {}
        
        # Create a quick-access category map
        scenarios_by_category = {}
        for scenario in scenarios:
            category = scenario['category']
            if category not in scenarios_by_category:
                scenarios_by_category[category] = []
            scenarios_by_category[category].append(scenario)
        
        # Create combined scenario for each category
        for category, category_scenarios in scenarios_by_category.items():
            # Skip if only one scenario in category
            if len(category_scenarios) <= 1:
                continue
                
            # Create combined scenario name and description
            name = f"Combined {category.title()} Optimization"
            description = f"Implement multiple {category} optimization strategies together"
            
            # Sum up savings with diminishing returns
            total_energy_savings = 0
            total_material_savings = 0
            total_investment = 0
            
            # Track individual scenarios used
            component_scenarios = []
            
            # Sort scenarios by ROI
            sorted_scenarios = sorted(
                category_scenarios,
                key=lambda s: scenario_results[s['id']]['roi_percentage'],
                reverse=True
            )
            
            # Add scenarios with diminishing returns
            diminishing_factor = 1.0
            for scenario in sorted_scenarios:
                result = scenario_results[scenario['id']]
                
                # Apply diminishing returns to savings
                energy_savings = result['annual_energy_savings'] * diminishing_factor
                material_savings = result['annual_material_savings'] * diminishing_factor
                
                total_energy_savings += energy_savings
                total_material_savings += material_savings
                total_investment += result['investment_cost']
                
                component_scenarios.append(scenario['id'])
                
                # Reduce factor for next scenario
                diminishing_factor *= 0.85  # 15% reduction for each additional scenario
            
            # Calculate financial metrics
            total_savings = total_energy_savings + total_material_savings
            if total_savings > 0:
                roi = (total_savings / total_investment) * 100
                payback_months = (total_investment / total_savings) * 12
                
                # Simple NPV calculation
                discount_rate = 0.10
                npv = -total_investment
                for year in range(1, 6):
                    npv += total_savings / ((1 + discount_rate) ** year)
            else:
                roi = 0
                payback_months = float('inf')
                npv = -total_investment
            
            # Store combined scenario results
            combined_scenarios[f"combined_{category}"] = {
                'name': name,
                'description': description,
                'component_scenarios': component_scenarios,
                'investment_cost': float(total_investment),
                'annual_energy_savings': float(total_energy_savings),
                'annual_material_savings': float(total_material_savings),
                'total_annual_savings': float(total_energy_savings + total_material_savings),
                'roi_percentage': float(roi),
                'payback_months': float(payback_months),
                'npv_5year': float(npv),
                'implementation_difficulty': 'high'  # Combined scenarios are usually more complex
            }
        
        # Create one overall combined scenario with best ROI options
        best_scenarios = sorted(
            scenario_results.items(),
            key=lambda item: item[1]['roi_percentage'],
            reverse=True
        )[:5]  # Top 5 scenarios by ROI
        
        total_energy_savings = 0
        total_material_savings = 0
        total_investment = 0
        component_scenarios = []
        
        # Apply diminishing returns
        diminishing_factor = 1.0
        for scenario_id, result in best_scenarios:
            # Apply diminishing returns to savings
            energy_savings = result['annual_energy_savings'] * diminishing_factor
            material_savings = result['annual_material_savings'] * diminishing_factor
            
            total_energy_savings += energy_savings
            total_material_savings += material_savings
            total_investment += result['investment_cost']
            
            component_scenarios.append(scenario_id)
            
            # Reduce factor for next scenario
            diminishing_factor *= 0.85
        
        # Calculate financial metrics
        total_savings = total_energy_savings + total_material_savings
        if total_savings > 0:
            roi = (total_savings / total_investment) * 100
            payback_months = (total_investment / total_savings) * 12
            
            # Simple NPV calculation
            discount_rate = 0.10
            npv = -total_investment
            for year in range(1, 6):
                npv += total_savings / ((1 + discount_rate) ** year)
        else:
            roi = 0
            payback_months = float('inf')
            npv = -total_investment
        
        # Store overall combined scenario
        combined_scenarios['combined_optimal'] = {
            'name': 'Optimal Combined Strategy',
            'description': 'Combination of highest ROI optimization strategies across all categories',
            'component_scenarios': component_scenarios,
            'investment_cost': float(total_investment),
            'annual_energy_savings': float(total_energy_savings),
            'annual_material_savings': float(total_material_savings),
            'total_annual_savings': float(total_energy_savings + total_material_savings),
            'roi_percentage': float(roi),
            'payback_months': float(payback_months),
            'npv_5year': float(npv),
            'implementation_difficulty': 'high'
        }
        
        return combined_scenarios
    
    def generate_reports(self, output_dir: str):
        """
        Generate reports for what-if analysis
        
        Args:
            output_dir: Directory to save the reports
        """
        # Create a subdirectory for what-if reports
        what_if_dir = os.path.join(output_dir, 'what_if_analysis')
        os.makedirs(what_if_dir, exist_ok=True)
        
        # Generate scenario comparison chart
        self._generate_scenario_comparison_chart(what_if_dir)
        
        # Generate ROI comparison chart
        self._generate_roi_comparison_chart(what_if_dir)
        
        # Generate combined scenarios chart
        self._generate_combined_scenarios_chart(what_if_dir)
        
        # Generate detailed report for each scenario
        self._generate_detailed_scenario_reports(what_if_dir)
    
    def _generate_scenario_comparison_chart(self, output_dir: str):
        """
        Generate comparison chart for individual scenarios
        
        Args:
            output_dir: Directory to save the chart
        """
        if not self.results or 'individual_scenarios' not in self.results:
            return
        
        individual_scenarios = self.results['individual_scenarios']
        
        # Extract data for plotting
        scenarios = []
        energy_savings = []
        material_savings = []
        
        for scenario_id, result in individual_scenarios.items():
            scenarios.append(result['name'])
            energy_savings.append(result['annual_energy_savings'])
            material_savings.append(result['annual_material_savings'])
        
        # Create chart
        plt.figure(figsize=(12, 6))
        
        # Set the color scheme as requested
        bar_width = 0.4
        x = np.arange(len(scenarios))
        
        # Create stacked bars
        plt.bar(x, energy_savings, bar_width, color='#FF9800', label='Energy Savings ($)')
        plt.bar(x, material_savings, bar_width, bottom=energy_savings, color='#9E9E9E', label='Material Savings ($)')
        
        plt.xlabel('Scenario')
        plt.ylabel('Annual Savings ($)')
        plt.title('Annual Savings by Scenario')
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Set the background color
        plt.gca().set_facecolor('white')
        plt.gcf().set_facecolor('white')
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'scenario_comparison.png'))
        plt.close()
    
    def _generate_roi_comparison_chart(self, output_dir: str):
        """
        Generate ROI comparison chart
        
        Args:
            output_dir: Directory to save the chart
        """
        if not self.results or 'individual_scenarios' not in self.results:
            return
        
        individual_scenarios = self.results['individual_scenarios']
        
        # Extract data for plotting
        scenarios = []
        roi_values = []
        payback_months = []
        
        for scenario_id, result in individual_scenarios.items():
            scenarios.append(result['name'])
            roi_values.append(result['roi_percentage'])
            payback_months.append(result['payback_months'])
        
        # Sort by ROI
        sorted_indices = np.argsort(roi_values)[::-1]  # Descending order
        scenarios = [scenarios[i] for i in sorted_indices]
        roi_values = [roi_values[i] for i in sorted_indices]
        payback_months = [payback_months[i] for i in sorted_indices]
        
        # Create chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Set the color scheme
        x = np.arange(len(scenarios))
        
        # ROI bars
        ax1.bar(x, roi_values, 0.4, color='#FF9800')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('ROI (%)', color='#FF9800')
        ax1.tick_params(axis='y', labelcolor='#FF9800')
        
        # Payback line
        ax2 = ax1.twinx()
        ax2.plot(x, payback_months, 'o-', color='#424242', linewidth=2, markersize=8)
        ax2.set_ylabel('Payback Period (Months)', color='#424242')
        ax2.tick_params(axis='y', labelcolor='#424242')
        
        plt.title('ROI and Payback Period by Scenario')
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.tight_layout()
        
        # Set the background color
        ax1.set_facecolor('white')
        fig.set_facecolor('white')
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'roi_comparison.png'))
        plt.close()
    
    def _generate_combined_scenarios_chart(self, output_dir: str):
        """
        Generate chart for combined scenarios
        
        Args:
            output_dir: Directory to save the chart
        """
        if not self.results or 'combined_scenarios' not in self.results:
            return
        
        combined_scenarios = self.results['combined_scenarios']
        
        # Extract data for plotting
        scenarios = []
        total_savings = []
        investments = []
        
        for scenario_id, result in combined_scenarios.items():
            scenarios.append(result['name'])
            total_savings.append(result['total_annual_savings'])
            investments.append(result['investment_cost'])
        
        # Create chart
        plt.figure(figsize=(12, 6))
        
        # Set the color scheme
        bar_width = 0.4
        x = np.arange(len(scenarios))
        
        # Create grouped bars
        plt.bar(x - bar_width/2, total_savings, bar_width, color='#FF9800', label='Annual Savings ($)')
        plt.bar(x + bar_width/2, investments, bar_width, color='#9E9E9E', label='Investment ($)')
        
        plt.xlabel('Scenario')
        plt.ylabel('Amount ($)')
        plt.title('Combined Scenarios: Investment vs. Annual Savings')
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Set the background color
        plt.gca().set_facecolor('white')
        plt.gcf().set_facecolor('white')
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'combined_scenarios.png'))
        plt.close()
    
    def _generate_detailed_scenario_reports(self, output_dir: str):
        """
        Generate detailed reports for each scenario
        
        Args:
            output_dir: Directory to save the reports
        """
        if not self.results:
            return
        
        # Create a text report for all scenarios
        with open(os.path.join(output_dir, 'scenarios_report.txt'), 'w') as f:
            f.write("FERRO ALLOY PLANT OPTIMIZATION SCENARIOS\n")
            f.write("======================================\n\n")
            
            # Individual scenarios
            if 'individual_scenarios' in self.results:
                f.write("INDIVIDUAL OPTIMIZATION SCENARIOS\n")
                f.write("-------------------------------\n\n")
                
                # Sort scenarios by ROI
                sorted_scenarios = sorted(
                    self.results['individual_scenarios'].items(),
                    key=lambda item: item[1]['roi_percentage'],
                    reverse=True
                )
                
                for scenario_id, result in sorted_scenarios:
                    f.write(f"Scenario: {result['name']}\n")
                    f.write(f"Implementation Difficulty: {result['implementation_difficulty'].title()}\n")
                    f.write(f"Investment Required: ${result['investment_cost']:,.2f}\n")
                    f.write(f"Annual Energy Savings: ${result['annual_energy_savings']:,.2f}\n")
                    f.write(f"Annual Material Savings: ${result['annual_material_savings']:,.2f}\n")
                    f.write(f"Total Annual Savings: ${result['total_annual_savings']:,.2f}\n")
                    f.write(f"ROI: {result['roi_percentage']:.1f}%\n")
                    f.write(f"Payback Period: {result['payback_months']:.1f} months\n")
                    f.write(f"5-Year NPV: ${result['npv_5year']:,.2f}\n")
                    
                    # Add scenario-specific actions if available
                    for scenario in self.scenarios:
                        if scenario['id'] == scenario_id and 'actions' in scenario:
                            f.write("Implementation Actions:\n")
                            for action in scenario['actions']:
                                f.write(f"- {action}\n")
                    
                    f.write("\n" + "-"*40 + "\n\n")
            
            # Combined scenarios
            if 'combined_scenarios' in self.results:
                f.write("\nCOMBINED OPTIMIZATION STRATEGIES\n")
                f.write("------------------------------\n\n")
                
                # Sort combined scenarios by ROI
                sorted_combined = sorted(
                    self.results['combined_scenarios'].items(),
                    key=lambda item: item[1]['roi_percentage'],
                    reverse=True
                )
                
                for scenario_id, result in sorted_combined:
                    f.write(f"Strategy: {result['name']}\n")
                    f.write(f"Description: {result.get('description', '')}\n")
                    f.write(f"Implementation Difficulty: {result['implementation_difficulty'].title()}\n")
                    f.write(f"Investment Required: ${result['investment_cost']:,.2f}\n")
                    f.write(f"Annual Energy Savings: ${result['annual_energy_savings']:,.2f}\n")
                    f.write(f"Annual Material Savings: ${result['annual_material_savings']:,.2f}\n")
                    f.write(f"Total Annual Savings: ${result['total_annual_savings']:,.2f}\n")
                    f.write(f"ROI: {result['roi_percentage']:.1f}%\n")
                    f.write(f"Payback Period: {result['payback_months']:.1f} months\n")
                    f.write(f"5-Year NPV: ${result['npv_5year']:,.2f}\n")
                    
                    # List component scenarios
                    if 'component_scenarios' in result:
                        f.write("Component Scenarios:\n")
                        for comp_id in result['component_scenarios']:
                            for scenario in self.scenarios:
                                if scenario['id'] == comp_id:
                                    f.write(f"- {scenario['name']}\n")
                    
                    f.write("\n" + "-"*40 + "\n\n")
            
            f.write("\nEND OF SCENARIOS REPORT\n")
