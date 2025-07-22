import streamlit as st
import pandas as pd
import yaml
import os
import tempfile

from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine
from modules.visualization import DashboardGenerator

st.set_page_config(page_title="Ferro Alloy Plant Optimization", layout="wide")

st.title("Ferro Alloy Plant Optimization System")

# --- Upload Section ---
st.sidebar.header("Upload Data Files (CSV)")
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
    uploaded_files[fname] = st.sidebar.file_uploader(f"{label} ({fname})", type="csv")

st.sidebar.markdown("---")
config_file = st.sidebar.file_uploader("Upload Config (default.yaml, optional)", type="yaml")

# --- Data Loading & Analysis ---
if st.button("Run Analysis"):
    # Save uploads to a temporary directory
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for fname, file in uploaded_files.items():
        if file:
            with open(os.path.join(data_dir, fname), "wb") as out:
                out.write(file.getbuffer())
    config_path = os.path.join(temp_dir, "config.yaml")
    if config_file:
        with open(config_path, "wb") as out:
            out.write(config_file.getbuffer())
    else:
        # Auto-generate config if not provided
        auto_config = {
            "data_files": {
                "energy_consumption": {"filename": "energy_consumption.csv", "date_columns": ["timestamp"]},
                "production": {"filename": "production.csv", "date_columns": ["timestamp"]},
                "material_input": {"filename": "material_input.csv", "date_columns": ["timestamp"]},
                "material_output": {"filename": "material_output.csv", "date_columns": ["timestamp"]},
                "furnace_data": {"filename": "furnace_data.csv", "date_columns": ["timestamp"]},
                "process_data": {"filename": "process_data.csv", "date_columns": ["timestamp"]},
            }
        }
        with open(config_path, "w") as out:
            yaml.dump(auto_config, out)
    # Load data & run analyses
    data_loader = DataLoader(data_dir=data_dir, config_path=config_path)
    datasets = data_loader.load_all_datasets()
    st.success("Data loaded!")
    baseline = BaselineAssessment(datasets)
    baseline_results = baseline.run_assessment()
    energy = EnergyAnalysis(datasets)
    energy_results = energy.analyze()
    material = MaterialAnalysis(datasets)
    material_results = material.analyze()
    furnace = FurnaceOptimization(datasets)
    furnace_results = furnace.optimize()
    electrode = ElectrodeOptimization(datasets)
    electrode_results = electrode.optimize()
    process = ProcessIntegration(datasets)
    process_results = process.optimize()
    whatif = WhatIfEngine(datasets)
    scenarios = whatif.generate_scenarios()
    whatif.run_scenarios(scenarios)
    dashboard = DashboardGenerator(
        baseline_results=baseline_results,
        energy_results=energy_results,
        material_results=material_results,
        furnace_results=furnace_results,
        electrode_results=electrode_results,
        process_results=process_results,
        what_if_scenarios=scenarios
    )
    dashboard.generate_dashboard(temp_dir)
    st.header("Analysis Dashboard")
    with open(f"{temp_dir}/dashboard/index.html") as f:
        st.components.v1.html(f.read(), height=900, scrolling=True)
else:
    st.info("Upload all required CSVs and click **Run Analysis**.")
