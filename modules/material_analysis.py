"""
Module for analyzing material flows and identifying optimization opportunities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import matplotlib.sankey as sankey
from scipy.stats import pearsonr

class MaterialAnalysis:
    """
    Analyzes material flows and identifies optimization opportunities
    """
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize the material analysis module
        
        Args:
            datasets: Dictionary of datasets loaded from CSV files
        """
        self.datasets = datasets
        self.results = {}
        
    def analyze(self):
        """
        Run the complete material analysis
        
        Returns:
            Dictionary with analysis results
        """
        # Check if we have material input and output data
        if 'material_input' not in self.datasets or 'material_output' not in self.datasets:
            self.results['error'] = "Material input/output data not available"
            return self.results
        
        # Analyze material balance
        self.results['material_balance'] = self._analyze_material_balance()
        
        # Analyze material yield
        self.results['material_yield'] = self._analyze_material_yield()
        
        # Analyze material loss by type
        self.results['material_loss'] = self._analyze_material_loss()
        
        # Analyze material cost
        self.results['material_cost'] = self._analyze_material_cost()
        
        # Analyze material quality
        self.results['material_quality'] = self._analyze_material_quality()
        
        # Analyze batch consistency
        if 'process_data' in self.datasets:
            self.results['batch_consistency'] = self._analyze_batch_consistency(self.datasets['process_data'])
        
        # Identify optimization opportunities
        self.results['optimization_opportunities'] = self._identify_optimization_opportunities()
        
        return self.results
    
    def _analyze_material_balance(self):
        """
        Analyze overall material balance
        
        Returns:
            Dictionary with material balance results
        """
        input_df = self.datasets['material_input']
        output_df = self.datasets['material_output']
        
        results = {}
        
        # Calculate total input and output
        total_input = input_df['quantity_tons'].sum()
        total_output = output_df['quantity_tons'].sum()
        
        results['total_input'] = float(total_input)
        results['total_output'] = float(total_output)
        results['material_loss'] = float(total_input - total_output)
        results['material_loss_percentage'] = float((total_input - total_output) / total_input * 100)
        
        # Analyze input by material type
        input_by_type = input_df.groupby('material_type')['quantity_tons'].sum().to_dict()
        results['input_by_type'] = {k: float(v) for k, v in input_by_type.items()}
        
        # Calculate input composition percentages
        input_composition = {k: float(v / total_input * 100) for k, v in input_by_type.items()}
        results['input_composition'] = input_composition
        
        # Analyze temporal patterns if we have timestamp data
        if 'timestamp' in input_df.columns:
            # Convert to datetime if necessary
            if not pd.api.types.is_datetime64_any_dtype(input_df['timestamp']):
                input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
            
            # Group by date
            input_df['date'] = input_df['timestamp'].dt.date
            daily_input = input_df.groupby('date')['quantity_tons'].sum()
            
            # Calculate daily statistics
            results['daily_input_avg'] = float(daily_input.mean())
            results['daily_input_min'] = float(daily_input.min())
            results['daily_input_max'] = float(daily_input.max())
            results['daily_input_std'] = float(daily_input.std())
        
        return results
    
    def _analyze_material_yield(self):
        """
        Analyze material yield
        
        Returns:
            Dictionary with material yield results
        """
        input_df = self.datasets['material_input']
        output_df = self.datasets['material_output']
        
        results = {}
        
        # Calculate overall yield
        total_input = input_df['quantity_tons'].sum()
        total_output = output_df['quantity_tons'].sum()
        
        overall_yield = total_output / total_input if total_input > 0 else 0
        results['overall_yield'] = float(overall_yield)
        results['overall_yield_percentage'] = float(overall_yield * 100)
        
        # Calculate yield over time if we have timestamp data
        if 'timestamp' in input_df.columns and 'timestamp' in output_df.columns:
            # Convert to datetime if necessary
            if not pd.api.types.is_datetime64_any_dtype(input_df['timestamp']):
                input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
            if not pd.api.types.is_datetime64_any_dtype(output_df['timestamp']):
                output_df['timestamp'] = pd.to_datetime(output_df['timestamp'])
            
            # Group by date
            input_df['date'] = input_df['timestamp'].dt.date
            output_df['date'] = output_df['timestamp'].dt.date
            
            daily_input = input_df.groupby('date')['quantity_tons'].sum().reset_index()
            daily_output = output_df.groupby('date')['quantity_tons'].sum().reset_index()
            
            # Merge daily input and output
            daily_yield = pd.merge(daily_input, daily_output, on='date', suffixes=('_input', '_output'))
            daily_yield['yield'] = daily_yield['quantity_tons_output'] / daily_yield['quantity_tons_input']
            
            # Calculate yield statistics
            results['daily_yield_avg'] = float(daily_yield['yield'].mean())
            results['daily_yield_min'] = float(daily_yield['yield'].min())
            results['daily_yield_max'] = float(daily_yield['yield'].max())
            results['daily_yield_std'] = float(daily_yield['yield'].std())
            
            # Identify days with best and worst yield
            best_day = daily_yield.loc[daily_yield['yield'].idxmax()]
            worst_day = daily_yield.loc[daily_yield['yield'].idxmin()]
            
            results['best_yield_day'] = {
                'date': str(best_day['date']),
                'yield': float(best_day['yield']),
                'input': float(best_day['quantity_tons_input']),
                'output': float(best_day['quantity_tons_output'])
            }
            
            results['worst_yield_day'] = {
                'date': str(worst_day['date']),
                'yield': float(worst_day['yield']),
                'input': float(worst_day['quantity_tons_input']),
                'output': float(worst_day['quantity_tons_output'])
            }
            
            # Calculate yield trend
            if len(daily_yield) > 2:
                # Simple linear regression for trend
                x = np.arange(len(daily_yield))
                y = daily_yield['yield'].values
                slope, intercept = np.polyfit(x, y, 1)
                results['yield_trend_slope'] = float(slope)
                results['yield_improving'] = bool(slope > 0)
        
        return results
    
    def _analyze_material_loss(self):
        """
        Analyze material losses
        
        Returns:
            Dictionary with material loss analysis
        """
        input_df = self.datasets['material_input']
        output_df = self.datasets['material_output']
        
        results = {}
        
        # Calculate overall loss
        total_input = input_df['quantity_tons'].sum()
        total_output = output_df['quantity_tons'].sum()
        
        total_loss = total_input - total_output
        loss_percentage = (total_loss / total_input * 100) if total_input > 0 else 0
        
        results['total_loss'] = float(total_loss)
        results['loss_percentage'] = float(loss_percentage)
        
        # Estimate loss distribution across process areas
        # This is an estimate as we don't have direct measurements of loss by area
        process_areas = ['Raw Material Handling', 'Batch Weighing', 'Conveyor Transport', 'Furnace', 'Metal-Slag Separation']
        
        # Assumed distribution based on typical ferro alloy plants
        loss_distribution = {
            'Raw Material Handling': 0.35,  # 35% of losses
            'Batch Weighing': 0.15,
            'Conveyor Transport': 0.25,
            'Furnace': 0.15,
            'Metal-Slag Separation': 0.10
        }
        
        # Calculate losses by area
        loss_by_area = {area: float(total_loss * dist) for area, dist in loss_distribution.items()}
        loss_percentage_by_area = {area: float(loss * 100 / total_input) if total_input > 0 else 0 
                                   for area, loss in loss_by_area.items()}
        
        results['loss_by_area'] = loss_by_area
        results['loss_percentage_by_area'] = loss_percentage_by_area
        
        # Calculate economic impact of losses
        if 'cost_per_ton' in input_df.columns:
            weighted_avg_cost = (input_df['quantity_tons'] * input_df['cost_per_ton']).sum() / total_input
            loss_value = total_loss * weighted_avg_cost
            results['loss_economic_value'] = float(loss_value)
            results['loss_economic_percentage'] = float(loss_value / (total_input * weighted_avg_cost) * 100)
        
        return results
    
    def _analyze_material_cost(self):
        """
        Analyze material costs
        
        Returns:
            Dictionary with material cost analysis
        """
        results = {}
        
        if 'material_input' in self.datasets and 'cost_per_ton' in self.datasets['material_input'].columns:
            input_df = self.datasets['material_input']
            
            # Calculate total and average material costs
            input_df['total_cost'] = input_df['quantity_tons'] * input_df['cost_per_ton']
            total_cost = input_df['total_cost'].sum()
            total_quantity = input_df['quantity_tons'].sum()
            avg_cost_per_ton = total_cost / total_quantity if total_quantity > 0 else 0
            
            results['total_material_cost'] = float(total_cost)
            results['avg_cost_per_ton'] = float(avg_cost_per_ton)
            
            # Cost by material type
            cost_by_type = input_df.groupby('material_type').agg({
                'total_cost': 'sum',
                'quantity_tons': 'sum'
            }).reset_index()
            
            cost_by_type['cost_per_ton'] = cost_by_type['total_cost'] / cost_by_type['quantity_tons']
            cost_by_type['percentage_of_total_cost'] = cost_by_type['total_cost'] / total_cost * 100
            
            results['cost_by_material_type'] = {
                row['material_type']: {
                    'total_cost': float(row['total_cost']),
                    'quantity_tons': float(row['quantity_tons']),
                    'cost_per_ton': float(row['cost_per_ton']),
                    'percentage_of_total_cost': float(row['percentage_of_total_cost'])
                }
                for _, row in cost_by_type.iterrows()
            }
            
            # Identify highest cost materials
            highest_cost_materials = cost_by_type.nlargest(3, 'cost_per_ton')
            results['highest_cost_materials'] = [
                {
                    'material_type': row['material_type'],
                    'cost_per_ton': float(row['cost_per_ton']),
                    'percentage_of_total_cost': float(row['percentage_of_total_cost'])
                }
                for _, row in highest_cost_materials.iterrows()
            ]
            
            # Calculate cost impact of yield improvements
            if 'material_output' in self.datasets:
                output_df = self.datasets['material_output']
                total_output = output_df['quantity_tons'].sum()
                current_yield = total_output / total_quantity if total_quantity > 0 else 0
                
                # Calculate cost savings for 1% yield improvement
                yield_improvement = 0.01  # 1%
                new_yield = min(current_yield + yield_improvement, 1.0)
                cost_savings = (new_yield - current_yield) * total_quantity * avg_cost_per_ton
                
                results['cost_savings_per_1pct_yield_improvement'] = float(cost_savings)
        
        return results
    
    def _analyze_material_quality(self):
        """
        Analyze material quality
        
        Returns:
            Dictionary with material quality analysis
        """
        results = {}
        
        if 'material_output' in self.datasets and 'quality_grade' in self.datasets['material_output'].columns:
            output_df = self.datasets['material_output']
            
            # Quality distribution
            quality_distribution = output_df.groupby('quality_grade')['quantity_tons'].sum()
            total_output = output_df['quantity_tons'].sum()
            
            quality_percentage = quality_distribution / total_output * 100 if total_output > 0 else 0
            results['quality_distribution'] = {
                grade: float(qty) for grade, qty in quality_distribution.items()
            }
            results['quality_percentage'] = {
                grade: float(pct) for grade, pct in quality_percentage.items()
            }
            
            # Quality trend over time
            if 'timestamp' in output_df.columns:
                # Convert to datetime if necessary
                if not pd.api.types.is_datetime64_any_dtype(output_df['timestamp']):
                    output_df['timestamp'] = pd.to_datetime(output_df['timestamp'])
                
                # Group by date and quality grade
                output_df['date'] = output_df['timestamp'].dt.date
                quality_by_date = output_df.pivot_table(
                    index='date',
                    columns='quality_grade',
                    values='quantity_tons',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Calculate percentage of each grade by date
                for col in quality_by_date.columns:
                    quality_by_date[f"{col}_pct"] = quality_by_date[col] / quality_by_date.sum(axis=1) * 100
                
                # Check if quality is improving over time
                if 'A' in quality_by_date.columns and len(quality_by_date) > 2:
                    # Simple linear regression for A grade trend
                    x = np.arange(len(quality_by_date))
                    if f"A_pct" in quality_by_date.columns:
                        y = quality_by_date['A_pct'].values
                        slope, intercept = np.polyfit(x, y, 1)
                        results['quality_trend_slope'] = float(slope)
                        results['quality_improving'] = bool(slope > 0)
        
        return results
    
    def _analyze_batch_consistency(self, process_df: pd.DataFrame):
        """
        Analyze batch consistency
        
        Args:
            process_df: Process data dataframe
            
        Returns:
            Dictionary with batch consistency analysis
        """
        results = {}
        
        if 'batch_weight_actual' in process_df.columns and 'batch_weight_target' in process_df.columns:
            # Calculate deviation from target
            process_df['weight_deviation'] = process_df['batch_weight_actual'] - process_df['batch_weight_target']
            process_df['weight_deviation_pct'] = (process_df['weight_deviation'] / process_df['batch_weight_target']) * 100
            
            # Calculate absolute deviation
            process_df['abs_deviation'] = process_df['weight_deviation'].abs()
            process_df['abs_deviation_pct'] = process_df['weight_deviation_pct'].abs()
            
            # Calculate statistics
            results['avg_deviation'] = float(process_df['weight_deviation'].mean())
            results['avg_abs_deviation'] = float(process_df['abs_deviation'].mean())
            results['avg_deviation_pct'] = float(process_df['weight_deviation_pct'].mean())
            results['avg_abs_deviation_pct'] = float(process_df['abs_deviation_pct'].mean())
            results['max_abs_deviation'] = float(process_df['abs_deviation'].max())
            results['max_abs_deviation_pct'] = float(process_df['abs_deviation_pct'].max())
            
            # Calculate percentage of batches within tolerance
            within_1pct = (process_df['abs_deviation_pct'] <= 1).mean() * 100
            within_2pct = (process_df['abs_deviation_pct'] <= 2).mean() * 100
            within_5pct = (process_df['abs_deviation_pct'] <= 5).mean() * 100
            
            results['within_1pct_tolerance'] = float(within_1pct)
            results['within_2pct_tolerance'] = float(within_2pct)
            results['within_5pct_tolerance'] = float(within_5pct)
            
            # Analyze consistency by process area
            if 'process_area' in process_df.columns:
                by_area = process_df.groupby('process_area').agg({
                    'abs_deviation_pct': 'mean',
                    'weight_deviation_pct': 'mean',
                    'batch_weight_actual': 'count'
                }).reset_index()
                
                by_area = by_area.rename(columns={'batch_weight_actual': 'batch_count'})
                
                results['consistency_by_area'] = {
                    row['process_area']: {
                        'avg_abs_deviation_pct': float(row['abs_deviation_pct']),
                        'avg_deviation_pct': float(row['weight_deviation_pct']),
                        'batch_count': int(row['batch_count'])
                    }
                    for _, row in by_area.iterrows()
                }
                
                # Identify most inconsistent area
                most_inconsistent = by_area.loc[by_area['abs_deviation_pct'].idxmax()]
                results['most_inconsistent_area'] = {
                    'area': most_inconsistent['process_area'],
                    'avg_abs_deviation_pct': float(most_inconsistent['abs_deviation_pct']),
                    'batch_count': int(most_inconsistent['batch_count'])
                }
            
            # Analyze trend in consistency
            if 'timestamp' in process_df.columns:
                # Convert to datetime if necessary
                if not pd.api.types.is_datetime64_any_dtype(process_df['timestamp']):
                    process_df['timestamp'] = pd.to_datetime(process_df['timestamp'])
                
                # Group by date
                process_df['date'] = process_df['timestamp'].dt.date
                daily_consistency = process_df.groupby('date')['abs_deviation_pct'].mean().reset_index()
                
                # Check for trend
                if len(daily_consistency) > 2:
                    x = np.arange(len(daily_consistency))
                    y = daily_consistency['abs_deviation_pct'].values
                    slope, intercept = np.polyfit(x, y, 1)
                    results['consistency_trend_slope'] = float(slope)
                    results['consistency_improving'] = bool(slope < 0)  # Lower deviation is better
        
        return results
    
    def _identify_optimization_opportunities(self):
        """
        Identify material optimization opportunities
        
        Returns:
            Dictionary with optimization opportunities
        """
        opportunities = []
        
        # Check material yield
        if 'material_yield' in self.results and 'overall_yield' in self.results['material_yield']:
            yield_pct = self.results['material_yield']['overall_yield'] * 100
            if yield_pct < 90:
                opportunities.append({
                    'area': 'Material Yield',
                    'current_value': float(yield_pct),
                    'target_value': 90.0,
                    'potential_improvement': float(90 - yield_pct),
                    'priority': 'high' if yield_pct < 85 else 'medium',
                    'description': 'Improve overall material yield through process optimization'
                })
        
        # Check material loss by area
        if 'material_loss' in self.results and 'loss_percentage_by_area' in self.results['material_loss']:
            for area, loss_pct in self.results['material_loss']['loss_percentage_by_area'].items():
                if loss_pct > 2.0:
                    opportunities.append({
                        'area': area,
                        'current_value': float(loss_pct),
                        'target_value': 2.0,
                        'potential_improvement': float(loss_pct - 2.0),
                        'priority': 'high' if loss_pct > 5.0 else 'medium',
                        'description': f'Reduce material loss in {area}'
                    })
        
        # Check batch consistency
        if 'batch_consistency' in self.results and 'avg_abs_deviation_pct' in self.results['batch_consistency']:
            dev_pct = self.results['batch_consistency']['avg_abs_deviation_pct']
            if dev_pct > 2.0:
                opportunities.append({
                    'area': 'Batch Consistency',
                    'current_value': float(dev_pct),
                    'target_value': 2.0,
                    'potential_improvement': float(dev_pct - 2.0),
                    'priority': 'high' if dev_pct > 5.0 else 'medium',
                    'description': 'Improve batch weighing accuracy and consistency'
                })
            
            # Check specific areas with poor consistency
            if 'consistency_by_area' in self.results['batch_consistency']:
                for area, stats in self.results['batch_consistency']['consistency_by_area'].items():
                    if stats['avg_abs_deviation_pct'] > 3.0:
                        opportunities.append({
                            'area': f'{area} Consistency',
                            'current_value': float(stats['avg_abs_deviation_pct']),
                            'target_value': 2.0,
                            'potential_improvement': float(stats['avg_abs_deviation_pct'] - 2.0),
                            'priority': 'medium',
                            'description': f'Improve batch consistency in {area}'
                        })
        
        # Check material quality
        if 'material_quality' in self.results and 'quality_percentage' in self.results['material_quality']:
            quality_pcts = self.results['material_quality']['quality_percentage']
            if 'A' in quality_pcts and quality_pcts['A'] < 80:
                opportunities.append({
                    'area': 'Product Quality',
                    'current_value': float(quality_pcts['A']),
                    'target_value': 80.0,
                    'potential_improvement': float(80 - quality_pcts['A']),
                    'priority': 'medium',
                    'description': 'Increase percentage of Grade A product'
                })
        
        # Calculate potential financial impact
        if 'material_cost' in self.results and 'cost_savings_per_1pct_yield_improvement' in self.results['material_cost']:
            savings_per_pct = self.results['material_cost']['cost_savings_per_1pct_yield_improvement']
            
            for i, opp in enumerate(opportunities):
                if 'potential_improvement' in opp:
                    # Estimate financial impact based on improvement potential
                    if 'Material Yield' in opp['area']:
                        financial_impact = opp['potential_improvement'] * savings_per_pct
                    else:
                        # Rough estimate for other areas
                        financial_impact = opp['potential_improvement'] * savings_per_pct * 0.5
                    
                    opportunities[i]['estimated_financial_impact'] = float(financial_impact)
        
        return opportunities
    
    def generate_reports(self, output_dir: str):
        """
        Generate reports and visualizations for material analysis
        
        Args:
            output_dir: Directory to save the reports
        """
        # Create a subdirectory for material reports
        material_dir = os.path.join(output_dir, 'material_analysis')
        os.makedirs(material_dir, exist_ok=True)
        
        # Generate material balance charts
        if 'material_balance' in self.results:
            self._generate_material_balance_charts(material_dir)
        
        # Generate material yield charts
        if 'material_yield' in self.results:
            self._generate_material_yield_charts(material_dir)
        
        # Generate material loss charts
        if 'material_loss' in self.results:
            self._generate_material_loss_charts(material_dir)
        
        # Generate material quality charts
        if 'material_quality' in self.results:
            self._generate_material_quality_charts(material_dir)
        
        # Generate batch consistency charts
        if 'batch_consistency' in self.results:
            self._generate_batch_consistency_charts(material_dir)
        
        # Generate optimization opportunity charts
        if 'optimization_opportunities' in self.results:
            self._generate_optimization_charts(material_dir)
        
        # Generate summary report
        self._generate_summary_report(material_dir)
    
    def _generate_material_balance_charts(self, output_dir: str):
        """Generate charts for material balance"""
        if 'material_balance' in self.results:
            balance = self.results['material_balance']
            
            if 'input_by_type' in balance:
                # Create a pie chart for material input composition
                plt.figure(figsize=(10, 6))
                materials = list(balance['input_by_type'].keys())
                quantities = list(balance['input_by_type'].values())
                
                plt.pie(quantities, labels=materials, autopct='%1.1f%%', startangle=90,
                       colors=plt.cm.Oranges(np.linspace(0.35, 0.65, len(materials))))
                plt.title('Material Input Composition')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'material_input_composition.png'))
                plt.close()
            
            # Create a Sankey diagram for material flow
            plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 1, 1)
            
            # Create a simple sankey diagram
            sankey_values = [balance.get('total_input', 0), balance.get('total_output', 0), 
                            balance.get('material_loss', 0)]
            
            # Use a simplified approach since matplotlib's Sankey is complex
            # Just create a horizontal bar chart showing input, output, and loss
            bars = ax.barh(['Input', 'Output', 'Loss'], sankey_values, color=['#FF9800', '#9E9E9E', '#424242'])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 1
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f} tons',
                       va='center')
            
            plt.title('Material Flow Overview')
            plt.xlabel('Quantity (tons)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'material_flow_overview.png'))
            plt.close()
    
    def _generate_material_yield_charts(self, output_dir: str):
        """Generate charts for material yield"""
        # Implementation would follow similar pattern to other visualization methods
        pass
    
    def _generate_material_loss_charts(self, output_dir: str):
        """Generate charts for material loss"""
        # Implementation would follow similar pattern to other visualization methods
        pass
    
    def _generate_material_quality_charts(self, output_dir: str):
        """Generate charts for material quality"""
        # Implementation would follow similar pattern to other visualization methods
        pass
    
    def _generate_batch_consistency_charts(self, output_dir: str):
        """Generate charts for batch consistency"""
        # Implementation would follow similar pattern to other visualization methods
        pass
    
    def _generate_optimization_charts(self, output_dir: str):
        """Generate charts for optimization opportunities"""
        # Implementation would follow similar pattern to other visualization methods
        pass
    
    def _generate_summary_report(self, output_dir: str):
        """Generate a summary report of material analysis findings"""
        # Implementation would follow similar pattern to other summary reports
        pass
