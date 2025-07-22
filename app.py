"""
Ferro Alloy Plant Optimization System - Main Application
Streamlit interface for data upload, analysis and visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine
from modules.visualization import DashboardGenerator

# Set page configuration
st.set_page_config(
    page_title="Ferro Alloy Plant Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color palette
COLOR_PALETTE = {
    'primary': '#FF9800',    # Orange
    'secondary': '#9E9E9E',  # Grey
    'tertiary': '#424242',   # Dark Grey
    'background': '#FFFFFF', # White
    'text': '#212121',       # Very Dark Grey
}

# Custom CSS
st.markdown(f"""
    <style>
    .main {{background-color: {COLOR_PALETTE['background']}}}
    .stApp {{background-color: {COLOR_PALETTE['background']}}}
    h1, h2, h3 {{color: {COLOR_PALETTE['tertiary']}}}
    .stButton>button {{
        background-color: {COLOR_PALETTE['primary']};
        color: white;
    }}
    .stProgress > div > div > div > div {{
        background-color: {COLOR_PALETTE['primary']};
    }}
    </style>
""", unsafe_allow_html=True)

# Page header
st.title("Ferro Alloy Plant Optimization System")
st.write("Upload plant data files to analyze energy and material efficiency and identify optimization opportunities.")

# Create sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/manufacturing.png", width=80)
st.sidebar.title("Carbon Resources")
st.sidebar.write("Digital Energy & Materials Analytics")
st.sidebar.markdown("---")

# Create tabs for different sections of the app
tab1, tab2, tab3 = st.tabs(["Data Upload", "Analysis & Results", "What-If Scenarios"])

# Global variables to store analysis results
datasets = {}
baseline_results = {}
energy_results = {}
material_results = {}
furnace_results = {}
electrode_results = {}
process_results = {}
scenarios = []
analysis_complete = False

# Create a temporary directory for file uploads
temp_dir = tempfile.mkdtemp()
data_dir = os.path.join(temp_dir, "data")
output_dir = os.path.join(temp_dir, "output")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

with tab1:
    st.header("Data Upload")
    
    # Use sample data option
    use_sample = st.checkbox("Use sample data for demonstration")
    
    if use_sample:
        st.success("Sample data will be used for analysis")
        # Copy sample data files from sample_data directory
        import shutil
        sample_dir = "sample_data"
        if os.path.exists(sample_dir):
            for file in os.listdir(sample_dir):
                if file.endswith(".csv"):
                    shutil.copy(os.path.join(sample_dir, file), os.path.join(data_dir, file))
    else:
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Energy Data")
            energy_file = st.file_uploader("Upload Energy Consumption Data", type="csv", 
                                          help="CSV with columns: timestamp, kwh_consumed, process_area, power_factor")
            production_file = st.file_uploader("Upload Production Data", type="csv",
                                             help="CSV with columns: timestamp, production_tons, product_type")
        
        with col2:
            st.subheader("Material Data")
            material_input_file = st.file_uploader("Upload Material Input Data", type="csv",
                                                 help="CSV with columns: timestamp, material_type, quantity_tons")
            material_output_file = st.file_uploader("Upload Material Output Data", type="csv",
                                                  help="CSV with columns: timestamp, product_type, quantity_tons")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Process Data")
            furnace_file = st.file_uploader("Upload Furnace Data", type="csv",
                                          help="CSV with columns: timestamp, temperature, power_factor, etc.")
            
        with col4:
            process_file = st.file_uploader("Upload Process Data", type="csv",
                                          help="CSV with columns: timestamp, process_area, uptime_hours, etc.")
        
        # Upload config file (optional)
        st.subheader("Configuration (Optional)")
        config_file = st.file_uploader("Upload Configuration File", type="yaml")
        
        # Save uploaded files
        if energy_file:
            with open(os.path.join(data_dir, "energy_consumption.csv"), "wb") as f:
                f.write(energy_file.getbuffer())
        
        if production_file:
            with open(os.path.join(data_dir, "production.csv"), "wb") as f:
                f.write(production_file.getbuffer())
                
        if material_input_file:
            with open(os.path.join(data_dir, "material_input.csv"), "wb") as f:
                f.write(material_input_file.getbuffer())
                
        if material_output_file:
            with open(os.path.join(data_dir, "material_output.csv"), "wb") as f:
                f.write(material_output_file.getbuffer())
                
        if furnace_file:
            with open(os.path.join(data_dir, "furnace_data.csv"), "wb") as f:
                f.write(furnace_file.getbuffer())
                
        if process_file:
            with open(os.path.join(data_dir, "process_data.csv"), "wb") as f:
                f.write(process_file.getbuffer())
        
        if config_file:
            with open(os.path.join(temp_dir, "config.yaml"), "wb") as f:
                f.write(config_file.getbuffer())
            config_path = os.path.join(temp_dir, "config.yaml")
        else:
            # Use default config
            config_path = "config/default.yaml"
            if not os.path.exists(config_path):
                # Create default config if it doesn't exist
                os.makedirs("config", exist_ok=True)
                with open(config_path, "w") as f:
                    yaml.dump({
                        "data_files": {
                            "energy_consumption": {"filename": "energy_consumption.csv", "date_columns": ["timestamp"]},
                            "production": {"filename": "production.csv", "date_columns": ["timestamp"]},
                            "material_input": {"filename": "material_input.csv", "date_columns": ["timestamp"]},
                            "material_output": {"filename": "material_output.csv", "date_columns": ["timestamp"]},
                            "furnace_data": {"filename": "furnace_data.csv", "date_columns": ["timestamp"]},
                            "process_data": {"filename": "process_data.csv", "date_columns": ["timestamp"]}
                        }
                    }, f)
    
    # Run analysis button
    if st.button("Run Analysis", key="run_analysis"):
        # Check if files exist
        required_files = [
            "energy_consumption.csv",
            "production.csv",
            "material_input.csv",
            "material_output.csv",
            "furnace_data.csv",
            "process_data.csv"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                missing_files.append(file)
        
        if missing_files and not use_sample:
            st.error(f"Missing required files: {', '.join(missing_files)}")
        else:
            # Load configuration
            if not os.path.exists(config_path):
                st.error("Configuration file not found.")
            else:
                with st.spinner("Loading data..."):
                    # Initialize data loader
                    data_loader = DataLoader(data_dir=data_dir, config_path=config_path)
                    datasets = data_loader.load_all_datasets()
                    
                    if not datasets:
                        st.error("Failed to load datasets. Check the data files.")
                    else:
                        # Run analyses
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        progress_text.text("Running baseline assessment...")
                        baseline = BaselineAssessment(datasets)
                        baseline_results = baseline.run_assessment()
                        progress_bar.progress(16)
                        
                        progress_text.text("Analyzing energy consumption...")
                        energy = EnergyAnalysis(datasets)
                        energy_results = energy.analyze()
                        progress_bar.progress(32)
                        
                        progress_text.text("Analyzing material flows...")
                        material = MaterialAnalysis(datasets)
                        material_results = material.analyze()
                        progress_bar.progress(48)
                        
                        progress_text.text("Optimizing furnace operations...")
                        furnace = FurnaceOptimization(datasets)
                        furnace_results = furnace.optimize()
                        progress_bar.progress(64)
                        
                        progress_text.text("Optimizing electrode operations...")
                        electrode = ElectrodeOptimization(datasets)
                        electrode_results = electrode.optimize()
                        progress_bar.progress(80)
                        
                        progress_text.text("Optimizing process integration...")
                        process = ProcessIntegration(datasets)
                        process_results = process.optimize()
                        progress_bar.progress(96)
                        
                        progress_text.text("Running what-if scenarios...")
                        what_if = WhatIfEngine(datasets)
                        scenarios = what_if.generate_scenarios()
                        what_if.run_scenarios(scenarios)
                        progress_bar.progress(100)
                        
                        progress_text.text("Generating reports and dashboards...")
                        # Generate dashboard
                        dashboard = DashboardGenerator(
                            baseline_results=baseline_results,
                            energy_results=energy_results,
                            material_results=material_results,
                            furnace_results=furnace_results,
                            electrode_results=electrode_results,
                            process_results=process_results,
                            what_if_scenarios=scenarios
                        )
                        dashboard.generate_dashboard(output_dir)
                        
                        st.session_state["analysis_complete"] = True
                        progress_text.text("Analysis complete!")
                        
                        # Switch to the Analysis tab
                        st.experimental_rerun()

with tab2:
    st.header("Analysis Results")
    
    if "analysis_complete" in st.session_state and st.session_state["analysis_complete"]:
        # Create subtabs for different analysis results
        subtab1, subtab2, subtab3, subtab4 = st.tabs(["Energy", "Material", "Process", "Summary"])
        
        with subtab1:
            st.subheader("Energy Analysis")
            
            # Create metrics for key findings
            if energy_results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'consumption_patterns' in energy_results and 'average_consumption' in energy_results['consumption_patterns']:
                        st.metric("Average Energy Consumption", 
                                 f"{energy_results['consumption_patterns']['average_consumption']:.2f} kWh")
                
                with col2:
                    if 'power_factor_analysis' in energy_results and 'average_pf' in energy_results['power_factor_analysis']:
                        st.metric("Average Power Factor", 
                                 f"{energy_results['power_factor_analysis']['average_pf']:.2f}",
                                 f"{(energy_results['power_factor_analysis']['average_pf'] - 0.95) * 100:.1f}%" if 'average_pf' in energy_results['power_factor_analysis'] else None)
                
                with col3:
                    if 'energy_intensity' in energy_results and 'average_intensity' in energy_results['energy_intensity']:
                        st.metric("Energy Intensity", 
                                 f"{energy_results['energy_intensity']['average_intensity']:.2f} kWh/ton")
            
            # Display energy charts
            st.image(os.path.join(output_dir, "energy_analysis", "hourly_consumption.png"), 
                    use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(output_dir, "energy_analysis", "energy_intensity_trend.png"), 
                        use_column_width=True)
            with col2:
                st.image(os.path.join(output_dir, "energy_analysis", "power_factor_trend.png"), 
                        use_column_width=True)
        
        with subtab2:
            st.subheader("Material Analysis")
            
            # Create metrics for key findings
            if material_results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'material_yield' in material_results:
                        st.metric("Material Yield", 
                                 f"{material_results['material_yield'] * 100:.1f}%")
                
                with col2:
                    if 'material_loss_percentage' in material_results:
                        st.metric("Material Loss", 
                                 f"{material_results['material_loss_percentage']:.1f}%",
                                 delta=-0.5, delta_color="inverse")
                
                with col3:
                    if 'specific_material_value' in material_results:
                        st.metric("Specific Material Value", 
                                 f"${material_results['specific_material_value']:.2f}/ton")
            
            # Display material flow diagrams
            st.image(os.path.join(output_dir, "material_analysis", "material_flow_sankey.png"), 
                    use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(output_dir, "material_analysis", "material_yield_pie.png"), 
                        use_column_width=True)
            with col2:
                st.image(os.path.join(output_dir, "material_analysis", "material_input_by_type.png"), 
                        use_column_width=True)
        
        with subtab3:
            st.subheader("Process Analysis")
            
            # Create metrics for key findings
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if furnace_results and 'temperature_analysis' in furnace_results:
                    st.metric("Average Temperature", 
                             f"{furnace_results['temperature_analysis'].get('mean_temperature', 0):.1f}°C")
            
            with col2:
                if electrode_results and 'paste_consumption' in electrode_results:
                    st.metric("Electrode Paste Consumption", 
                             f"{electrode_results['paste_consumption'].get('specific_consumption', 0):.2f} kg/ton")
            
            with col3:
                if process_results and 'equipment_utilization' in process_results:
                    st.metric("Equipment Utilization", 
                             f"{process_results['equipment_utilization'].get('overall_utilization', 0) * 100:.1f}%")
            
            # Display process optimization charts
            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(output_dir, "furnace_optimization", "temperature_stability.png"), 
                        use_column_width=True)
            with col2:
                st.image(os.path.join(output_dir, "process_integration", "process_efficiency.png"), 
                        use_column_width=True)
        
        with subtab4:
            st.subheader("Summary Dashboard")
            
            # Load the HTML dashboard
            dashboard_path = os.path.join(output_dir, "dashboard", "index.html")
            if os.path.exists(dashboard_path):
                with open(dashboard_path, "r") as f:
                    dashboard_html = f.read()
                st.components.v1.html(dashboard_html, height=800, scrolling=True)
            else:
                st.error("Dashboard not found. Please run the analysis first.")
            
            # Download button for the dashboard
            with open(dashboard_path, "rb") as file:
                st.download_button(
                    label="Download Dashboard HTML",
                    data=file,
                    file_name="ferro_alloy_dashboard.html",
                    mime="text/html"
                )
    else:
        st.info("Please run the analysis from the 'Data Upload' tab first.")

with tab3:
    st.header("What-If Scenarios")
    
    if "analysis_complete" in st.session_state and st.session_state["analysis_complete"]:
        # Create filtering options
        st.subheader("Filter Scenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_filter = st.multiselect(
                "Category",
                ["energy", "material", "furnace", "electrode", "process"],
                default=["energy", "material", "furnace"]
            )
        
        with col2:
            difficulty_filter = st.multiselect(
                "Implementation Difficulty",
                ["low", "medium", "high"],
                default=["low", "medium"]
            )
        
        # Display filtered scenarios
        st.subheader("Optimization Opportunities")
        
        # Filter scenarios based on selection
        filtered_scenarios = [s for s in scenarios if 
                             s.get('category', '') in category_filter and 
                             s.get('implementation_difficulty', '') in difficulty_filter]
        
        if filtered_scenarios:
            # Sort scenarios by ROI
            sorted_scenarios = sorted(filtered_scenarios, 
                                     key=lambda x: x.get('energy_savings_percentage', 0) + x.get('material_savings_percentage', 0),
                                     reverse=True)
            
            for i, scenario in enumerate(sorted_scenarios):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {i+1}. {scenario.get('name', 'Unnamed Scenario')}")
                    st.markdown(f"**{scenario.get('description', '')}**")
                    
                    # Display key metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Energy Savings", f"{scenario.get('energy_savings_percentage', 0):.1f}%")
                    with metrics_col2:
                        st.metric("Material Savings", f"{scenario.get('material_savings_percentage', 0):.1f}%")
                    with metrics_col3:
                        st.metric("Cost Savings", f"{scenario.get('cost_savings_percentage', 0):.1f}%")
                    
                    # Display implementation actions
                    if 'actions' in scenario:
                        st.markdown("**Implementation Actions:**")
                        for action in scenario['actions']:
                            st.markdown(f"- {action}")
                
                with col2:
                    # Display ROI and difficulty
                    st.markdown(f"**Category:** {scenario.get('category', '').title()}")
                    st.markdown(f"**Difficulty:** {scenario.get('implementation_difficulty', '').title()}")
                    st.markdown(f"**Investment:** {scenario.get('investment_level', '').title()}")
                    st.markdown(f"**Payback:** {scenario.get('payback_period_months', 0):.1f} months")
                
                st.markdown("---")
        else:
            st.warning("No scenarios match the selected filters.")
        
        # Display ROI comparison chart
        st.subheader("Return on Investment Comparison")
        st.image(os.path.join(output_dir, "what_if_analysis", "roi_comparison.png"), 
                use_column_width=True)
    else:
        st.info("Please run the analysis from the 'Data Upload' tab first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.write("Created by: sandy1983jsr")
