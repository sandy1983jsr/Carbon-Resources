"""
Ferro Alloy Plant Analytics & Optimization
Main application file with Streamlit interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling with orange/grey/white theme
st.markdown("""
<style>
    .main {background-color: #FFFFFF;}
    .stApp {background-color: #FFFFFF;}
    h1 {color: #FF9800;}
    h2 {color: #424242;}
    h3 {color: #616161;}
    .stButton>button {
        background-color: #FF9800;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #FF9800;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Ferro Alloy Plant Analytics & Optimization")
st.markdown("### Digital energy and materials efficiency solution")

# Sidebar with upload options
st.sidebar.header("Upload Data Files")
st.sidebar.markdown("Upload your data files or use sample data")

# Create a temporary directory to store uploaded files
temp_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(temp_dir, "outputs"), exist_ok=True)

# Define required file uploads
required_files = [
    ("energy_consumption.csv", "Energy Consumption Data"),
    ("production.csv", "Production Data"),
    ("material_input.csv", "Material Input Data"),
    ("material_output.csv", "Material Output Data"),
    ("furnace_data.csv", "Furnace Operation Data"),
    ("process_data.csv", "Process Equipment Data")
]

# Create file upload widgets
uploaded_files = {}
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)

if use_sample_data:
    st.sidebar.success("Using sample data for demonstration")
    data_dir = "sample_data"
else:
    data_dir = temp_dir
    for file_name, file_desc in required_files:
        uploaded_file = st.sidebar.file_uploader(f"Upload {file_desc}", type=["csv"], key=file_name)
        
        if uploaded_file is not None:
            # Save the uploaded file to the temporary directory
            with open(os.path.join(temp_dir, file_name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files[file_name] = True
        else:
            uploaded_files[file_name] = False

# Config options
config_option = st.sidebar.selectbox(
    "Configuration",
    ["Default", "Custom"],
    index=0
)

if config_option == "Custom":
    uploaded_config = st.sidebar.file_uploader("Upload custom config", type=["yaml"])
    if uploaded_config is not None:
        config_path = os.path.join(temp_dir, "custom_config.yaml")
        with open(config_path, "wb") as f:
            f.write(uploaded_config.getbuffer())
    else:
        config_path = "config/default.yaml"
else:
    config_path = "config/default.yaml"

# Main analysis section
if st.sidebar.button("Run Analysis"):
    if use_sample_data or all(uploaded_files.values()):
        with st.spinner("Loading data..."):
            data_loader = DataLoader(data_dir=data_dir, config_path=config_path)
            datasets = data_loader.load_all_datasets()
            st.success("Data loaded successfully")
        
        # Progress bar for analysis
        progress_bar = st.progress(0)
        
        # Run analysis modules
        analyses = {}
        
        with st.spinner("Running baseline assessment..."):
            baseline = BaselineAssessment(datasets)
            analyses['baseline_results'] = baseline.run_assessment()
            progress_bar.progress(16)
        
        with st.spinner("Analyzing energy consumption..."):
            energy = EnergyAnalysis(datasets)
            analyses['energy_results'] = energy.analyze()
            progress_bar.progress(32)
        
        with st.spinner("Analyzing material flow..."):
            material = MaterialAnalysis(datasets)
            analyses['material_results'] = material.analyze()
            progress_bar.progress(48)
        
        with st.spinner("Optimizing furnace operation..."):
            furnace = FurnaceOptimization(datasets)
            analyses['furnace_results'] = furnace.optimize()
            progress_bar.progress(64)
        
        with st.spinner("Optimizing electrode performance..."):
            electrode = ElectrodeOptimization(datasets)
            analyses['electrode_results'] = electrode.optimize()
            progress_bar.progress(80)
        
        with st.spinner("Analyzing process integration..."):
            process = ProcessIntegration(datasets)
            analyses['process_results'] = process.optimize()
            progress_bar.progress(90)
        
        with st.spinner("Running what-if scenarios..."):
            what_if = WhatIfEngine(datasets)
            scenarios = what_if.generate_scenarios()
            what_if.run_scenarios(scenarios)
            analyses['what_if_results'] = what_if.results
            analyses['what_if_scenarios'] = scenarios
            progress_bar.progress(100)
        
        # Generate dashboard
        with st.spinner("Generating dashboards..."):
            dashboard = DashboardGenerator(**analyses)
            dashboard_path = os.path.join(temp_dir, "outputs", "dashboard")
            os.makedirs(dashboard_path, exist_ok=True)
            dashboard.generate_dashboard(dashboard_path)
        
        # Display results
        st.markdown("## Analysis Results")
        
        # Create tabs for different result sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Executive Summary", 
            "Energy Analysis", 
            "Material Analysis",
            "Optimization Scenarios"
        ])
        
        with tab1:
            st.header("Executive Summary")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                energy_intensity = analyses['energy_results'].get('energy_intensity', {}).get('avg_intensity', 0)
                st.metric("Energy Intensity", f"{energy_intensity:.1f} kWh/ton")
                
            with col2:
                material_yield = analyses['material_results'].get('yield', 0) * 100
                st.metric("Material Yield", f"{material_yield:.1f}%")
                
            with col3:
                power_factor = analyses['energy_results'].get('power_factor_analysis', {}).get('avg_pf', 0)
                st.metric("Avg Power Factor", f"{power_factor:.2f}")
            
            # Summary findings
            st.subheader("Key Findings")
            st.markdown("""
            - **Energy Savings Potential**: 7-15% through power factor correction, load balancing and furnace optimization
            - **Material Savings Potential**: 3-8% through improved weighing, conveyor efficiency and metal-slag separation
            - **Overall Cost Reduction**: Significant opportunities identified in electrode paste consumption and process integration
            """)
            
            # Display a key chart
            st.image(os.path.join(dashboard_path, "energy_efficiency_gauge.png"))
        
        with tab2:
            st.header("Energy Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(dashboard_path, "hourly_consumption.png"))
                st.image(os.path.join(dashboard_path, "power_factor_trend.png"))
            
            with col2:
                st.image(os.path.join(dashboard_path, "energy_intensity_trend.png"))
                
                # Energy savings table
                st.subheader("Energy Saving Opportunities")
                energy_opportunities = analyses['what_if_results']['individual_scenarios']
                energy_data = []
                
                for scenario_id, result in energy_opportunities.items():
                    if 'energy' in scenario_id:
                        energy_data.append({
                            'Opportunity': result['name'],
                            'Annual Savings (kWh)': f"{result['annual_energy_savings']:,.0f}",
                            'ROI': f"{result['roi_percentage']:.1f}%",
                            'Payback': f"{result['payback_months']:.1f} months"
                        })
                
                if energy_data:
                    st.table(pd.DataFrame(energy_data))
        
        with tab3:
            st.header("Material Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(dashboard_path, "material_flow_sankey.png"))
            
            with col2:
                st.image(os.path.join(dashboard_path, "material_yield_pie.png"))
                
                # Material savings table
                st.subheader("Material Saving Opportunities")
                material_opportunities = analyses['what_if_results']['individual_scenarios']
                material_data = []
                
                for scenario_id, result in material_opportunities.items():
                    if 'material' in scenario_id:
                        material_data.append({
                            'Opportunity': result['name'],
                            'Annual Savings ($)': f"{result['annual_material_savings']:,.0f}",
                            'ROI': f"{result['roi_percentage']:.1f}%",
                            'Payback': f"{result['payback_months']:.1f} months"
                        })
                
                if material_data:
                    st.table(pd.DataFrame(material_data))
        
        with tab4:
            st.header("Optimization Scenarios")
            
            st.image(os.path.join(dashboard_path, "optimization_opportunities.png"))
            st.image(os.path.join(dashboard_path, "roi_comparison.png"))
            
            # Combined scenarios
            st.subheader("Combined Optimization Strategies")
            if 'combined_scenarios' in analyses['what_if_results']:
                combined = analyses['what_if_results']['combined_scenarios']
                combined_data = []
                
                for scenario_id, result in combined.items():
                    combined_data.append({
                        'Strategy': result['name'],
                        'Annual Savings': f"${result['total_annual_savings']:,.0f}",
                        'Investment': f"${result['investment_cost']:,.0f}",
                        'ROI': f"{result['roi_percentage']:.1f}%",
                        'Payback': f"{result['payback_months']:.1f} months"
                    })
                
                if combined_data:
                    st.table(pd.DataFrame(combined_data))
            
        # Download section
        st.markdown("### Download Results")
        
        with open(os.path.join(dashboard_path, "index.html"), "r") as f:
            dashboard_html = f.read()
        
        st.download_button(
            label="Download Full Dashboard (HTML)",
            data=dashboard_html,
            file_name="ferro_alloy_dashboard.html",
            mime="text/html"
        )
        
        # Display the report date and time
        st.markdown(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    else:
        st.error("Please upload all required data files or use sample data")

# Display instructions when first loading
else:
    st.info("### How to use this tool")
    st.markdown("""
    1. Choose whether to use sample data or upload your own files
    2. If using your own data, upload all required CSV files
    3. Select configuration option (Default or Custom)
    4. Click "Run Analysis" to process the data
    5. View results in the interactive dashboard
    6. Download the full report for offline viewing
    
    This tool will analyze energy consumption, material flow, and process efficiency to identify optimization opportunities.
    """)
    
    # Display sample of expected data format
    with st.expander("CSV File Format Examples"):
        st.markdown("**energy_consumption.csv** example:")
        st.code("""timestamp,kwh_consumed,process_area,power_factor
2023-01-01 00:00:00,2450,Furnace,0.91
2023-01-01 01:00:00,2380,Furnace,0.92""")
        
        st.markdown("**furnace_data.csv** example:")
        st.code("""timestamp,temperature,power_factor,electrode_current,electrode_paste_consumption_kg,energy_consumed,production_tons
2023-01-01 00:00:00,1650,0.91,18500,120,2450,5.2
2023-01-01 01:00:00,1645,0.92,18450,118,2380,5.1""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"© 2025 Carbon Resources Digital Analytics")
st.sidebar.markdown(f"User: sandy1983jsr")
st.sidebar.markdown(f"Last updated: 2025-07-22")
