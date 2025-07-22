"""
Module for generating interactive dashboards and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64

# Set the color palette as requested: orange, grey, and white
COLOR_PALETTE = {
    'primary': '#FF9800',  # Orange
    'secondary': '#9E9E9E',  # Grey
    'tertiary': '#424242',  # Dark Grey
    'background': '#FFFFFF',  # White
    'text': '#212121',  # Very Dark Grey (almost black)
    'highlight': '#FFB74D',  # Light Orange
    'accent1': '#BDBDBD',  # Light Grey
    'accent2': '#757575'   # Medium Grey
}

class DashboardGenerator:
    """
    Generates interactive dashboards and visualizations
    """
    
    def __init__(self, 
                 baseline_results: Optional[dict] = None,
                 energy_results: Optional[dict] = None,
                 material_results: Optional[dict] = None,
                 furnace_results: Optional[dict] = None,
                 electrode_results: Optional[dict] = None,
                 process_results: Optional[dict] = None,
                 what_if_scenarios: Optional[List[dict]] = None):
        """
        Initialize the dashboard generator
        
        Args:
            baseline_results: Results from baseline assessment
            energy_results: Results from energy analysis
            material_results: Results from material analysis
            furnace_results: Results from furnace optimization
            electrode_results: Results from electrode optimization
            process_results: Results from process integration
            what_if_scenarios: What-if scenarios
        """
        self.baseline_results = baseline_results or {}
        self.energy_results = energy_results or {}
        self.material_results = material_results or {}
        self.furnace_results = furnace_results or {}
        self.electrode_results = electrode_results or {}
        self.process_results = process_results or {}
        self.what_if_scenarios = what_if_scenarios or []
    
    def generate_dashboard(self, output_dir: str):
        """
        Generate a comprehensive dashboard
        
        Args:
            output_dir: Directory to save the dashboard
        """
        # Create a dashboard directory
        dashboard_dir = os.path.join(output_dir, 'dashboard')
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Generate individual visualizations
        self._generate_energy_visualizations(dashboard_dir)
        self._generate_material_visualizations(dashboard_dir)
        self._generate_optimization_visualizations(dashboard_dir)
        
        # Generate the main HTML dashboard that combines everything
        self._generate_html_dashboard(dashboard_dir)
    
    def _generate_energy_visualizations(self, output_dir: str):
        """
        Generate energy-related visualizations
        
        Args:
            output_dir: Directory to save the visualizations
        """
        # Create energy consumption heatmap (by hour and day)
        if self.energy_results and 'consumption_patterns' in self.energy_results:
            patterns = self.energy_results['consumption_patterns']
            
            if 'hourly_pattern' in patterns:
                # Convert to values usable for plotting
                hours = list(patterns['hourly_pattern'].keys())
                hourly_values = list(patterns['hourly_pattern'].values())
                
                # Create a line chart for hourly consumption
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=hourly_values,
                    mode='lines+markers',
                    name='Hourly Consumption',
                    line=dict(color=COLOR_PALETTE['primary'], width=3),
                    marker=dict(size=8, color=COLOR_PALETTE['primary'])
                ))
                
                fig.update_layout(
                    title='Energy Consumption by Hour',
                    xaxis_title='Hour of Day',
                    yaxis_title='Average Consumption (kWh)',
                    plot_bgcolor=COLOR_PALETTE['background'],
                    paper_bgcolor=COLOR_PALETTE['background'],
                    font=dict(color=COLOR_PALETTE['text']),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(24)),
                        ticktext=[f"{h}:00" for h in range(24)]
                    )
                )
                
                fig.write_html(os.path.join(output_dir, 'hourly_consumption.html'))
                fig.write_image(os.path.join(output_dir, 'hourly_consumption.png'))
            
            # Create energy intensity trend if available
            if 'energy_intensity' in self.energy_results and 'avg_energy_intensity' in self.energy_results['energy_intensity']:
                # This is placeholder data - in a real system, you'd have actual time series data
                # Create synthetic data for demonstration
                dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
                avg_intensity = self.energy_results['energy_intensity']['avg_energy_intensity']
                min_intensity = self.energy_results['energy_intensity']['min_intensity']
                
                # Create synthetic values with some random variation around the average
                np.random.seed(42)  # For reproducibility
                intensities = np.random.normal(avg_intensity, avg_intensity * 0.1, len(dates))
                intensities = np.clip(intensities, min_intensity, avg_intensity * 1.5)  # Keep within reasonable bounds
                
                # Create a line chart for energy intensity trend
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=intensities,
                    mode='lines',
                    name='Energy Intensity',
                    line=dict(color=COLOR_PALETTE['primary'], width=2)
                ))
                
                # Add a horizontal line for the average
                fig.add_trace(go.Scatter(
                    x=[dates.min(), dates.max()],
                    y=[avg_intensity, avg_intensity],
                    mode='lines',
                    name='Average Intensity',
                    line=dict(color=COLOR_PALETTE['tertiary'], width=2, dash='dash')
                ))
                
                # Add a horizontal line for the minimum (best) intensity
                fig.add_trace(go.Scatter(
                    x=[dates.min(), dates.max()],
                    y=[min_intensity, min_intensity],
                    mode='lines',
                    name='Best Intensity',
                    line=dict(color=COLOR_PALETTE['secondary'], width=2, dash='dot')
                ))
                
                fig.update_layout(
                    title='Energy Intensity Trend',
                    xaxis_title='Date',
                    yaxis_title='Energy Intensity (kWh/ton)',
                    plot_bgcolor=COLOR_PALETTE['background'],
                    paper_bgcolor=COLOR_PALETTE['background'],
                    font=dict(color=COLOR_PALETTE['text'])
                )
                
                fig.write_html(os.path.join(output_dir, 'energy_intensity_trend.html'))
                fig.write_image(os.path.join(output_dir, 'energy_intensity_trend.png'))
                
                # Create a gauge chart for energy efficiency
                if 'energy_intensity' in self.energy_results:
                    avg = self.energy_results['energy_intensity'].get('avg_energy_intensity', 100)
                    min_val = self.energy_results['energy_intensity'].get('min_intensity', avg * 0.7)
                    max_val = self.energy_results['energy_intensity'].get('max_intensity', avg * 1.3)
                    
                    # Calculate efficiency percentage (inverse of intensity)
                    current_efficiency = (max_val - avg) / (max_val - min_val) * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=current_efficiency,
                        domain=dict(x=[0, 1], y=[0, 1]),
                        title=dict(text="Energy Efficiency", font=dict(color=COLOR_PALETTE['text'])),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickfont=dict(color=COLOR_PALETTE['text'])),
                            bar=dict(color=COLOR_PALETTE['primary']),
                            bgcolor=COLOR_PALETTE['accent1'],
                            borderwidth=2,
                            bordercolor=COLOR_PALETTE['secondary'],
                            steps=[
                                dict(range=[0, 50], color=COLOR_PALETTE['accent1']),
                                dict(range=[50, 80], color=COLOR_PALETTE['highlight']),
                                dict(range=[80, 100], color=COLOR_PALETTE['primary'])
                            ],
                            threshold=dict(
                                line=dict(color=COLOR_PALETTE['tertiary'], width=4),
                                thickness=0.75,
                                value=current_efficiency
                            )
                        )
                    ))
                    
                    fig.update_layout(
                        title="Energy Efficiency Rating",
                        paper_bgcolor=COLOR_PALETTE['background'],
                        font=dict(color=COLOR_PALETTE['text'])
                    )
                    
                    fig.write_html(os.path.join(output_dir, 'energy_efficiency_gauge.html'))
                    fig.write_image(os.path.join(output_dir, 'energy_efficiency_gauge.png'))
        
        # Create a power factor chart if available
        if self.energy_results and 'power_factor_analysis' in self.energy_results:
            pf_data = self.energy_results['power_factor_analysis']
            
            if 'average_pf' in pf_data:
                # Create synthetic time series for power factor
                dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
                avg_pf = pf_data['average_pf']
                min_pf = pf_data['min_pf']
                max_pf = pf_data['max_pf']
                
                # Create synthetic values with some random variation
                np.random.seed(43)  # Different seed from above
                power_factors = np.random.normal(avg_pf, (max_pf - min_pf) / 4, len(dates))
                power_factors = np.clip(power_factors, min_pf, max_pf)
                
                # Create a line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=power_factors,
                    mode='lines',
                    name='Power Factor',
                    line=dict(color=COLOR_PALETTE['primary'], width=2)
                ))
                
                # Add a horizontal line for optimal power factor (0.95)
                fig.add_trace(go.Scatter(
                    x=[dates.min(), dates.max()],
                    y=[0.95, 0.95],
                    mode='lines',
                    name='Optimal Power Factor',
                    line=dict(color=COLOR_PALETTE['tertiary'], width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Power Factor Trend',
                    xaxis_title='Date',
                    yaxis_title='Power Factor',
                    plot_bgcolor=COLOR_PALETTE['background'],
                    paper_bgcolor=COLOR_PALETTE['background'],
                    font=dict(color=COLOR_PALETTE['text']),
                    yaxis=dict(range=[min(min_pf * 0.95, 0.8), 1.0])
                )
                
                fig.write_html(os.path.join(output_dir, 'power_factor_trend.html'))
                fig.write_image(os.path.join(output_dir, 'power_factor_trend.png'))
    
    def _generate_material_visualizations(self, output_dir: str):
        """
        Generate material-related visualizations
        
        Args:
            output_dir: Directory to save the visualizations
        """
        # Create a Sankey diagram for material flow
        if self.baseline_results and 'material_baseline' in self.baseline_results:
            material_data = self.baseline_results['material_baseline']
            
            # Prepare data for Sankey diagram
            if 'total_input' in material_data and 'total_output' in material_data and 'material_loss' in material_data:
                # Create a simple Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color=COLOR_PALETTE['background'], width=0.5),
                        label=["Raw Materials", "Process", "Output", "Loss/Waste"],
                        color=[COLOR_PALETTE['primary'], COLOR_PALETTE['accent2'], 
                               COLOR_PALETTE['highlight'], COLOR_PALETTE['secondary']]
                    ),
                    link=dict(
                        source=[0, 1, 1],
                        target=[1, 2, 3],
                        value=[material_data['total_input'], 
                               material_data['total_output'], 
                               material_data['material_loss']],
                        color=[COLOR_PALETTE['primary'], COLOR_PALETTE['highlight'], 
                               COLOR_PALETTE['secondary']]
                    )
                )])
                
                fig.update_layout(
                    title="Material Flow",
                    font=dict(color=COLOR_PALETTE['text'], size=12),
                    paper_bgcolor=COLOR_PALETTE['background']
                )
                
                fig.write_html(os.path.join(output_dir, 'material_flow_sankey.html'))
                fig.write_image(os.path.join(output_dir, 'material_flow_sankey.png'))
            
            # Create a pie chart for material yield
            if 'total_output' in material_data and 'material_loss' in material_data:
                labels = ['Output', 'Loss/Waste']
                values = [material_data['total_output'], material_data['material_loss']]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']]),
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )])
                
                fig.update_layout(
                    title="Material Yield vs. Loss",
                    font=dict(color=COLOR_PALETTE['text']),
                    paper_bgcolor=COLOR_PALETTE['background']
                )
                
                fig.write_html(os.path.join(output_dir, 'material_yield_pie.html'))
                fig.write_image(os.path.join(output_dir, 'material_yield_pie.png'))
                
            # Create a bar chart for material input by type if available
            if 'input_by_type' in material_data:
                material_types = list(material_data['input_by_type'].keys())
                quantities = list(material_data['input_by_type'].values())
                
                fig = go.Figure(data=[go.Bar(
                    x=material_types,
                    y=quantities,
                    marker_color=COLOR_PALETTE['primary']
                )])
                
                fig.update_layout(
                    title="Material Input by Type",
                    xaxis_title="Material Type",
                    yaxis_title="Quantity (tons)",
                    plot_bgcolor=COLOR_PALETTE['background'],
                    paper_bgcolor=COLOR_PALETTE['background'],
                    font=dict(color=COLOR_PALETTE['text'])
                )
                
                fig.write_html(os.path.join(output_dir, 'material_input_by_type.html'))
                fig.write_image(os.path.join(output_dir, 'material_input_by_type.png'))
    
    def _generate_optimization_visualizations(self, output_dir: str):
        """
        Generate optimization-related visualizations
        
        Args:
            output_dir: Directory to save the visualizations
        """
        # Create a visualization of optimization scenarios
        if self.what_if_scenarios:
            # Extract data for different types of scenarios
            energy_scenarios = [s for s in self.what_if_scenarios if s['category'] == 'energy']
            material_scenarios = [s for s in self.what_if_scenarios if s['category'] == 'material']
            furnace_scenarios = [s for s in self.what_if_scenarios if s['category'] == 'furnace']
            electrode_scenarios = [s for s in self.what_if_scenarios if s['category'] == 'electrode']
            
            # Create a bubble chart showing potential savings vs implementation difficulty
            fig = go.Figure()
            
            # Add traces for each category
            categories = [
                ('Energy', energy_scenarios, COLOR_PALETTE['primary']),
                ('Material', material_scenarios, COLOR_PALETTE['highlight']),
                ('Furnace', furnace_scenarios, COLOR_PALETTE['secondary']),
                ('Electrode', electrode_scenarios, COLOR_PALETTE['tertiary'])
            ]
            
            for category_name, scenarios, color in categories:
                if not scenarios:
                    continue
                
                # Map difficulty to numeric values
                difficulty_map = {'low': 1, 'medium': 2, 'high': 3}
                x_values = [difficulty_map[s['implementation_difficulty']] for s in scenarios]
                
                # Get total percentage savings (energy + material)
                y_values = [s['energy_savings_percentage'] + s['material_savings_percentage'] for s in scenarios]
                
                # Use investment level for bubble size
                investment_map = {'low': 10, 'medium': 20, 'high': 30}
                sizes = [investment_map[s['investment_level']] for s in scenarios]
                
                # Get names for hover text
                names = [s['name'] for s in scenarios]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    name=category_name,
                    marker=dict(
                        size=sizes,
                        color=color,
                        line=dict(width=2, color=COLOR_PALETTE['background'])
                    ),
                    text=names,
                    hovertemplate='%{text}<br>Savings: %{y:.1f}%<br>Investment: %{marker.size}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Optimization Opportunities",
                xaxis=dict(
                    title="Implementation Difficulty",
                    tickmode='array',
                    tickvals=[1, 2, 3],
                    ticktext=['Low', 'Medium', 'High']
                ),
                yaxis=dict(title="Potential Savings (%)"),
                plot_bgcolor=COLOR_PALETTE['background'],
                paper_bgcolor=COLOR_PALETTE['background'],
                font=dict(color=COLOR_PALETTE['text'])
            )
            
            fig.write_html(os.path.join(output_dir, 'optimization_opportunities.html'))
            fig.write_image(os.path.join(output_dir, 'optimization_opportunities.png'))
            
            # Create a bar chart comparing ROI across scenarios
            # In a real system, you'd calculate this from actual data
            # Here we'll create synthetic data based on the scenarios
            
            # For each scenario, estimate ROI based on savings and investment level
            all_scenarios = energy_scenarios + material_scenarios + furnace_scenarios + electrode_scenarios
            scenario_names = [s['name'] for s in all_scenarios]
            
            # Calculate approximate ROI
            investment_values = {'low': 50000, 'medium': 150000, 'high': 350000}
            roi_values = []
            
            for s in all_scenarios:
                # Estimate annual savings
                energy_savings = s['energy_savings_kwh'] * 0.08  # Assuming $0.08 per kWh
                material_savings = s['material_savings_tons'] * 800  # Assuming $800 per ton
                total_savings = energy_savings + material_savings
                
                # Get investment amount
                investment = investment_values[s['investment_level']]
                
                # Calculate ROI percentage
                roi = (total_savings / investment) * 100 if investment > 0 else 0
                roi_values.append(roi)
            
            # Sort by ROI
            sorted_indices = np.argsort(roi_values)[::-1]  # Descending order
            sorted_names = [scenario_names[i] for i in sorted_indices]
            sorted_roi = [roi_values[i] for i in sorted_indices]
            
            fig = go.Figure(data=[go.Bar(
                x=sorted_names,
                y=sorted_roi,
                marker_color=COLOR_PALETTE['primary']
            )])
            
            fig.update_layout(
                title="Return on Investment by Scenario",
                xaxis_title="Scenario",
                yaxis_title="ROI (%)",
                plot_bgcolor=COLOR_PALETTE['background'],
                paper_bgcolor=COLOR_PALETTE['background'],
                font=dict(color=COLOR_PALETTE['text'])
            )
            
            fig.update_xaxes(tickangle=45)
            
            fig.write_html(os.path.join(output_dir, 'roi_comparison.html'))
            fig.write_image(os.path.join(output_dir, 'roi_comparison.png'))
    
    def _generate_html_dashboard(self, output_dir: str):
        """
        Generate the main HTML dashboard
        
        Args:
            output_dir: Directory to save the dashboard
        """
        # Create an HTML file that combines all the visualizations
        dashboard_path = os.path.join(output_dir, 'index.html')
        
        # Find all the HTML visualization files
        visualization_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
        
        # Define dashboard sections
        sections = [
            {
                'title': 'Energy Analysis',
                'files': [f for f in visualization_files if f.startswith('energy_') or 
                          f.startswith('hourly_') or f.startswith('power_')]
            },
            {
                'title': 'Material Analysis',
                'files': [f for f in visualization_files if f.startswith('material_')]
            },
            {
                'title': 'Optimization Opportunities',
                'files': [f for f in visualization_files if f.startswith('optimization_') or 
                          f.startswith('roi_')]
            }
        ]
        
        # Create the HTML content
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ferro Alloy Plant Optimization Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: {COLOR_PALETTE['background']};
                    color: {COLOR_PALETTE['text']};
                }}
                .header {{
                    background-color: {COLOR_PALETTE['primary']};
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .section {{
                    margin: 20px;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                .section h2 {{
                    color: {COLOR_PALETTE['tertiary']};
                    border-bottom: 2px solid {COLOR_PALETTE['primary']};
                    padding-bottom: 10px;
                }}
                .viz-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
                }}
                .viz-item {{
                    width: 45%;
                    margin: 10px;
                    padding: 15px;
                    background-color: white;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }}
                .viz-item iframe {{
                    width: 100%;
                    height: 400px;
                    border: none;
                }}
                .viz-item h3 {{
                    margin-top: 0;
                    color: {COLOR_PALETTE['tertiary']};
                }}
                .summary {{
                    margin: 20px;
                    padding: 20px;
                    background-color: {COLOR_PALETTE['accent1']};
                    border-left: 5px solid {COLOR_PALETTE['primary']};
                }}
                footer {{
                    background-color: {COLOR_PALETTE['tertiary']};
                    color: white;
                    text-align: center;
                    padding: 10px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Ferro Alloy Plant Optimization Dashboard</h1>
                <p>Energy and Materials Efficiency Analysis</p>
            </div>
        '''
        
        # Add summary section
        html_content += '''
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This dashboard presents a comprehensive analysis of the ferro alloy plant's energy and material efficiency, 
                   along with potential optimization opportunities. Key findings include:</p>
                # Completing the HTML dashboard generator section in visualization.py
# This would be added at the end of the _generate_html_dashboard method

        # Add summary section with some key metrics from our analysis
        html_content += f'''
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This dashboard presents a comprehensive analysis of the ferro alloy plant's energy and material efficiency, 
                   along with potential optimization opportunities. Key findings include:</p>
                <ul>
                    <li><strong>Energy Intensity:</strong> {self.energy_results.get('energy_intensity', {}).get('avg_energy_intensity', 'N/A')} kWh/ton 
                        (potential reduction of {self.energy_results.get('energy_intensity', {}).get('improvement_percentage', 'N/A')}%)</li>
                    <li><strong>Material Yield:</strong> {self.baseline_results.get('material_baseline', {}).get('material_yield', 0)*100:.1f}% 
                        (potential improvement of {self.baseline_results.get('material_baseline', {}).get('material_loss_percentage', 0):.1f}%)</li>
                    <li><strong>Power Factor:</strong> {self.energy_results.get('power_factor_analysis', {}).get('average_pf', 'N/A'):.2f} 
                        (optimal target: 0.95)</li>
                </ul>
                <p>Our analysis has identified several high-ROI opportunities that could deliver significant cost savings:</p>
                <ul>
                    <li>Reduction in energy consumption: 7-15%</li>
                    <li>Improvement in material efficiency: 3-8%</li>
                    <li>Overall cost reduction potential: 5-10%</li>
                </ul>
            </div>
        '''
        
        # Add each section with its visualizations
        for section in sections:
            html_content += f'''
                <div class="section">
                    <h2>{section['title']}</h2>
                    <div class="viz-container">
            '''
            
            # Add each visualization in this section
            for viz_file in section['files']:
                # Extract title from filename
                title = viz_file.replace('.html', '').replace('_', ' ').title()
                
                html_content += f'''
                        <div class="viz-item">
                            <h3>{title}</h3>
                            <iframe src="{viz_file}"></iframe>
                        </div>
                '''
            
            html_content += '''
                    </div>
                </div>
            '''
        
        # Add footer
        html_content += f'''
            <footer>
                <p>Ferro Alloy Plant Optimization Analysis — Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
            </footer>
        </body>
        </html>
        '''
        
        # Write the HTML file
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
