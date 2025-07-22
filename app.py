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
import modules.visualization as viz

st.set_page_config(page_title="Ferro Alloy Consulting Dashboard", layout="wide")
COLOR_PALETTE = ["#FF9800", "#9E9E9E", "#FFFFFF"]

st.markdown(
    """
    <style>
    .reportview-container {background-color: #FFFFFF;}
    .sidebar .sidebar-content {background-color: #FFFFFF;}
    h1 {color: #FF9800;}
    h2, h3 {color: #424242;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ferro Alloy Consulting Dashboard")
st.write("Upload operational CSVs to analyze energy, material, and cost opportunities.")

with st.sidebar:
    st.header("Upload Data Files")
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
        uploaded_files[fname] = st.file_uploader(f"{label} ({fname})", type="csv")
    config_file = st.file_uploader("Config file (optional)", type="yaml")

if st.button("Run Analysis"):
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
    data_loader = DataLoader(data_dir=data_dir, config_path=config_path)
    datasets = data_loader.load_all_datasets()
    st.success("Data loaded!")

    # --- Analyses ---
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

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Energy Dashboard", "Material Dashboard", "Process & Furnace", "Scenario/ROI"
    ])

    with tab1:
        st.header("⚡ Energy Dashboard")
        st.metric("Total Energy Consumption (kWh)", f"{baseline_results.get('energy_total',0):,.0f}")
        st.metric("Average Power Factor", f"{energy_results.get('avg_pf',0):.2f}")
        st.metric("Average Energy Intensity (kWh)", f"{energy_results.get('avg_intensity',0):.2f}")
        area_data = energy_results.get("area")
        if area_data is not None:
            fig = viz.plot_energy_area_bar(area_data)
            if fig is not None:
                st.plotly_chart(fig)
        hourly = energy_results.get("hourly")
        if hourly is not None:
            fig = viz.plot_energy_hourly(hourly)
            if fig is not None:
                st.plotly_chart(fig)
        daily = energy_results.get("daily")
        if daily is not None:
            fig = viz.plot_energy_trend(daily, anomalies=energy_results.get("anomaly_vals"))
            if fig is not None:
                st.plotly_chart(fig)

    with tab2:
        st.header("🧱 Material Dashboard")
        st.metric("Material Yield (%)", f"{100*(material_results.get('material_yield',0)):.2f}")
        st.metric("Material Loss (%)", f"{material_results.get('material_loss_percentage',0):.2f}")
        input_by_type = material_results.get("input_by_type")
        if input_by_type:
            fig = viz.plot_material_pie(input_by_type)
            if fig is not None:
                st.plotly_chart(fig)
        if material_results.get("material_yield") is not None and material_results.get("material_loss_percentage") is not None:
            fig = viz.plot_material_yield(material_results["material_yield"], material_results["material_loss_percentage"])
            if fig is not None:
                st.plotly_chart(fig)
        st.plotly_chart(viz.plot_sankey())

    with tab3:
        st.header("🔥 Process & Furnace Dashboard")
        st.metric("Furnace Mean Temp (°C)", f"{furnace_results.get('mean_temp',0):.0f}")
        st.metric("Furnace PF", f"{furnace_results.get('mean_pf',0):.2f}")
        if electrode_results.get("paste_consumption"):
            st.metric("Electrode Paste (kg/ton)", f"{electrode_results['paste_consumption'].get('specific_consumption',0):.2f}")
        if process_results.get("equipment_utilization"):
            st.metric("Equipment Utilization (%)", f"{100*process_results['equipment_utilization'].get('overall_utilization',0):.2f}")

    with tab4:
        st.header("🧮 Scenario & ROI Dashboard")
        fig = viz.plot_scenario_roi(scenarios)
        if fig is not None:
            st.plotly_chart(fig)
        for s in scenarios:
            st.subheader(f"Scenario: {s['name']}")
            st.write(s["description"])
            st.write(f"**Energy savings:** {s['energy_savings_percentage']}%")
            st.write(f"**Material savings:** {s['material_savings_percentage']}%")
            st.write(f"**Cost savings:** {s['cost_savings_percentage']}%")
            st.write(f"**Difficulty:** {s['implementation_difficulty']}")
            st.write(f"**Investment Level:** {s['investment_level']}")
            st.write(f"**Payback (months):** {s['payback_period_months']}")
            st.write("**Recommended actions:**")
            for action in s["actions"]:
                st.write(f"- {action}")

    st.success("Consulting dashboard complete. Use each tab to guide client discussion and action planning.")
else:
    st.info("Upload all required CSVs and click **Run Analysis**.")
