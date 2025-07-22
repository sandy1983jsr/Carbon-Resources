import streamlit as st
import pandas as pd
import yaml
from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine
from modules.visualization import DashboardGenerator

st.set_page_config(page_title="Ferro Alloy Optimization", layout="wide")

st.title("Ferro Alloy Plant Analytics & Optimization")

# --- Upload Section ---
st.sidebar.header("Upload Data Files")
uploaded_files = {}
required_files = [
    ("energy_consumption.csv", "Energy Consumption"),
    ("production.csv", "Production"),
    ("material_input.csv", "Material Input"),
    ("material_output.csv", "Material Output"),
    ("furnace_data.csv", "Furnace Data"),
    ("process_data.csv", "Process Data"),
]
for fname, label in required_files:
    uploaded_files[fname] = st.sidebar.file_uploader(f"Upload {label} ({fname})", type="csv")

# Config upload (optional)
config_file = st.sidebar.file_uploader("Upload Config (default.yaml)", type="yaml")

# Load configuration
if config_file:
    config = yaml.safe_load(config_file)
else:
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

# --- Data Loading ---
if st.sidebar.button("Run Analysis"):
    # Save uploaded files to disk for loader
    import os
    os.makedirs("tmp_uploads", exist_ok=True)
    for fname, file in uploaded_files.items():
        if file:
            with open(f"tmp_uploads/{fname}", "wb") as out:
                out.write(file.getbuffer())
    # Load data
    data_loader = DataLoader(data_dir="tmp_uploads", config_path="config/default.yaml")
    datasets = data_loader.load_all_datasets()
    st.success("Data loaded!")

    # --- Analysis ---
    with st.spinner("Running baseline assessment..."):
        baseline = BaselineAssessment(datasets)
        baseline_results = baseline.run_assessment()
    with st.spinner("Analyzing energy..."):
        energy = EnergyAnalysis(datasets)
        energy_results = energy.analyze()
    with st.spinner("Analyzing materials..."):
        material = MaterialAnalysis(datasets)
        material_results = material.analyze()
    with st.spinner("Optimizing furnace..."):
        furnace = FurnaceOptimization(datasets)
        furnace_results = furnace.optimize()
    with st.spinner("Electrode optimization..."):
        electrode = ElectrodeOptimization(datasets)
        electrode_results = electrode.optimize()
    with st.spinner("Process integration..."):
        process = ProcessIntegration(datasets)
        process_results = process.optimize()
    with st.spinner("What-if analysis..."):
        whatif = WhatIfEngine(datasets)
        scenarios = whatif.generate_scenarios()
        whatif.run_scenarios(scenarios)

    # --- Dashboard ---
    st.header("Analysis Dashboard")
    dashboard = DashboardGenerator(
        baseline_results=baseline_results,
        energy_results=energy_results,
        material_results=material_results,
        furnace_results=furnace_results,
        electrode_results=electrode_results,
        process_results=process_results,
        what_if_scenarios=scenarios
    )
    dashboard.generate_dashboard("tmp_uploads")
    st.markdown("View interactive dashboard below or [download HTML](tmp_uploads/dashboard/index.html)")
    with open("tmp_uploads/dashboard/index.html", "r") as f:
        html = f.read()
        st.components.v1.html(html, height=1200, scrolling=True)
else:
    st.info("Upload all required files and click **Run Analysis**.")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 Carbon Resources Digital Analytics Demo")
