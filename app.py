import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import yaml

# --- Import modules ---
from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine
import modules.visualization as visualization
import modules.recommendations as recommendations  # Function-based, not class-based

st.set_page_config(page_title="Ferro Alloy Consulting Dashboard", layout="wide")
st.title("Ferro Alloy Consulting Dashboard")
st.write("Upload operational CSVs or use random demo data to analyze energy, material, and cost opportunities.")

# --- Sidebar: Data source selection ---
with st.sidebar:
    st.header("Data Source")
    use_random = st.checkbox("Use random demo data (no file upload)", value=False)
    st.markdown("---")

    uploaded_files = {}
    if not use_random:
        st.header("Upload Data Files")
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
        st.markdown("---")
        st.info("Tip: For best results, upload all six files.")

        # Show preview of each CSV after upload
        for fname, file in uploaded_files.items():
            if file:
                st.markdown(f"**Preview: {fname}**")
                df = pd.read_csv(file)
                st.dataframe(df.head())
                file.seek(0)  # Reset pointer for actual loading

if st.button("Run Analysis"):
    if use_random:
        from modules.random_data import generate_random_datasets
        datasets = generate_random_datasets(n_hours=24*7)  # 7 days of hourly data
        st.success("Random demo data generated!")
        for k, df in datasets.items():
            st.write(f"**{k}** sample data:")
            st.dataframe(df.head())
    else:
        # Save uploaded files to temp directory
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        for fname, file in uploaded_files.items():
            if file:
                with open(os.path.join(data_dir, fname), "wb") as out:
                    out.write(file.getbuffer())
        config_path = os.path.join(temp_dir, "config.yaml")
        if 'config_file' in locals() and config_file:
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
        st.write("Loaded datasets:", list(datasets.keys()))
        for k, df in datasets.items():
            if df is not None:
                st.write(f"{k}: Shape {df.shape}")
                st.dataframe(df.head())
            else:
                st.warning(f"{k}: Not loaded")

    # If no data, stop
    if not datasets or all(df is None or df.empty for df in datasets.values()):
        st.error("No data loaded. Please upload all required CSV files or use random demo data.")
        st.stop()

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
    whatif_results = whatif.run_scenarios(scenarios)

    # --- Recommendations (function-based) ---
    if hasattr(recommendations, "generate"):
        recs = recommendations.generate(datasets)
    else:
        recs = []

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Energy", "Material", "Process & Furnace", "Furnace Optimization", "What-if", 
        "Recommendations", "Savings", "Validation", "Benchmarking & Actions"
    ])

    with tab1:
        st.header("⚡ Energy Dashboard")
        # Defensive checks for baseline_results and energy_results
        if isinstance(baseline_results, dict):
            total_energy = baseline_results.get('energy_total', 0)
            energy_intensity = baseline_results.get('energy_intensity_kwh_per_ton', 0)
        else:
            total_energy = 0
            energy_intensity = 0
        if isinstance(energy_results, dict):
            avg_pf = energy_results.get('avg_pf', 0)
        else:
            avg_pf = 0

        st.metric("Total Energy Consumption (kWh)", f"{total_energy:,.0f}")
        st.metric("Average Power Factor", f"{avg_pf:.2f}")
        st.metric("Energy Intensity (kWh/ton)", f"{energy_intensity:,.2f}")

        if isinstance(energy_results, dict) and "area" in energy_results and energy_results["area"] is not None:
            st.subheader("Energy by Area")
            st.dataframe(energy_results["area"])
            fig = visualization.plot_energy_area_bar(energy_results["area"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        if isinstance(energy_results, dict) and "hourly" in energy_results and energy_results["hourly"] is not None:
            st.subheader("Hourly Energy Consumption (kWh)")
            fig = visualization.plot_energy_hourly(energy_results["hourly"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        if isinstance(energy_results, dict) and "daily" in energy_results and energy_results["daily"] is not None:
            st.subheader("Daily Energy Consumption")
            fig = visualization.plot_energy_trend(energy_results["daily"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🧱 Material Dashboard")
        if isinstance(baseline_results, dict):
            material_input_total = baseline_results.get('material_input_total', 0)
            material_output_total = baseline_results.get('material_output_total', 0)
            material_yield = baseline_results.get('material_yield', 0)
            material_loss_pct = baseline_results.get('material_loss_pct', 0)
        else:
            material_input_total = 0
            material_output_total = 0
            material_yield = 0
            material_loss_pct = 0
        st.metric("Material Input (tons)", f"{material_input_total:.2f}")
        st.metric("Material Output (tons)", f"{material_output_total:.2f}")
        st.metric("Material Yield (%)", f"{100*(material_yield):.2f}")
        st.metric("Material Loss (%)", f"{material_loss_pct:.2f}")
        if isinstance(material_results, dict) and "input_by_type" in material_results and material_results["input_by_type"] is not None:
            st.subheader("Material Input Split")
            st.dataframe(pd.DataFrame.from_dict(material_results["input_by_type"], orient="index", columns=["tons"]))
            fig = visualization.plot_material_pie(material_results["input_by_type"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        fig = visualization.plot_material_yield(
            material_yield,
            material_loss_pct
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🔥 Process & Furnace Dashboard")
        if isinstance(furnace_results, dict):
            mean_temp = furnace_results.get('mean_temp', 0)
            mean_pf = furnace_results.get('mean_pf', 0)
            recommendations_list = furnace_results.get("recommendations", [])
        else:
            mean_temp = 0
            mean_pf = 0
            recommendations_list = []
        st.metric("Furnace Mean Temp (°C)", f"{mean_temp:.0f}")
        st.metric("Furnace PF", f"{mean_pf:.2f}")
        if isinstance(electrode_results, dict) and electrode_results.get("paste_consumption"):
            st.metric("Electrode Paste (kg/ton)", f"{electrode_results['paste_consumption'].get('specific_consumption',0):.2f}")
        if isinstance(process_results, dict) and process_results.get("equipment_utilization"):
            st.metric("Equipment Utilization (%)", f"{100*process_results['equipment_utilization'].get('overall_utilization',0):.2f}")
        st.subheader("Visualization")
        # Add furnace/process plots as needed

    with tab4:
        st.header("🔧 Furnace Optimization")
        if isinstance(furnace_results, dict) and recommendations_list:
            for rec in recommendations_list:
                st.success(rec)
        else:
            st.info("No furnace optimization recommendations available.")
        st.subheader("Visualization")
        # Add furnace optimization visualizations as needed

    with tab5:
        st.header("🔮 What-if Analysis")
        st.write("Scenarios:")
        if isinstance(scenarios, list):
            for s in scenarios:
                st.info(f"Scenario: {s.get('change','')} - Impact: {s.get('impact','')}")
        if isinstance(whatif_results, list) and len(whatif_results) > 0:
            st.write("Results:")
            st.dataframe(pd.DataFrame(whatif_results))
        else:
            st.info("No what-if analysis results available.")
        # Add what-if scenario visualizations as needed

    with tab6:
        st.header("🧑‍🔬 Recommendations")
        if recs:
            for rec in recs:
                st.success(rec)
        else:
            st.info("No recommendations generated.")

    with tab7:
        st.header("💲 Estimated Savings")
        st.info("Savings analysis module is currently unavailable.")

    with tab8:
        st.header("✔️ Data Validation")
        st.info("Validation module is currently unavailable.")

    with tab9:
        st.header("📊 Benchmarking")
        st.info("Benchmarking module is currently unavailable.")
        st.header("📝 Action Tracker")
        st.info("Action Tracker module is currently unavailable.")

    st.success("Analysis complete! Explore the tabs above for results.")

else:
    st.info("Upload all required CSVs or select random demo data, then click **Run Analysis**.")
