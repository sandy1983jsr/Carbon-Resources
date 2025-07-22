import streamlit as st
import tempfile
import os
import yaml

from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine

# ...other imports...

st.title("Consulting Dashboard")

with st.sidebar:
    st.header("Upload Data Files")
    uploaded_files = {
        fname: st.file_uploader(f"{label} ({fname})", type="csv")
        for fname, label in [
            ("energy_consumption.csv", "Energy Consumption"),
            ("production.csv", "Production"),
            ("material_input.csv", "Material Input"),
            ("material_output.csv", "Material Output"),
            ("furnace_data.csv", "Furnace Data"),
            ("process_data.csv", "Process Data"),
        ]
    }
    config_file = st.file_uploader("Config file (optional)", type="yaml")

if st.button("Run Analysis"):
    # Save uploaded files to a temp directory
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
        # Create a simple config if none uploaded
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

    # --- HERE IS THE KEY CHANGE: USE THE LOADER ---
    data_loader = DataLoader(data_dir=data_dir, config_path=config_path)
    datasets = data_loader.load_all_datasets()  # This is a dict of {name: DataFrame}

    # --- NOW PASS datasets TO MODULES ---
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

    # ...rest of your code, all modules use datasets from loader...

else:
    st.info("Upload all files and click Run Analysis.")
