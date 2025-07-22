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
import modules.recommendations as recommendations
import modules.savings_tracker as savings_tracker
import modules.validation as validation
import modules.benchmarking as benchmarking
import modules.action_tracker as action_tracker

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
        datasets = generate_random_datasets(n_hours=24)
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
    recs = recommendations.Recommendations(datasets).generate()
    savings_results = savings_tracker.SavingsTracker(datasets).calculate()
    validation_results = validation.Validation(datasets).validate()
    benchmarking_results = benchmarking.Benchmarking(datasets).compare()
    actions = action_tracker.ActionTracker().get_actions()

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Energy", "Material", "Process & Furnace", "Furnace Optimization", "What-if", 
        "Recommendations", "Savings", "Validation", "Benchmarking & Actions"
    ])

    with tab1:
        st.header("⚡ Energy Dashboard")
        st.metric("Total Energy Consumption (kWh)", f"{baseline_results.get('energy_total',0):,.0f}")
        st.metric("Average Power Factor", f"{energy_results.get('avg_pf',0):.2f}")
        st.metric("Energy Intensity (kWh/ton)", f"{baseline_results.get('energy_intensity_kwh_per_ton',0):.2f}")

        if "area" in energy_results and energy_results["area"] is not None:
            st.subheader("Energy by Area")
            st.dataframe(energy_results["area"])
            fig = visualization.plot_energy_area_bar(energy_results["area"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        if "hourly" in energy_results and energy_results["hourly"] is not None:
            st.subheader("Hourly Energy Consumption (kWh)")
            fig = visualization.plot_energy_hourly(energy_results["hourly"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        if "daily" in energy_results and energy_results["daily"] is not None:
            st.subheader("Daily Energy Consumption")
            fig = visualization.plot_energy_trend(energy_results["daily"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🧱 Material Dashboard")
        st.metric("Material Input (tons)", f"{baseline_results.get('material_input_total',0):.2f}")
        st.metric("Material Output (tons)", f"{baseline_results.get('material_output_total',0):.2f}")
        st.metric("Material Yield (%)", f"{100*(baseline_results.get('material_yield',0)):.2f}")
        st.metric("Material Loss (%)", f"{baseline_results.get('material_loss_pct',0):.2f}")
        if "input_by_type" in material_results and material_results["input_by_type"] is not None:
            st.subheader("Material Input Split")
            st.dataframe(pd.DataFrame.from_dict(material_results["input_by_type"], orient="index", columns=["tons"]))
            fig = visualization.plot_material_pie(material_results["input_by_type"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        fig = visualization.plot_material_yield(
            baseline_results.get('material_yield', None),
            baseline_results.get('material_loss_pct', None)
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🔥 Process & Furnace Dashboard")
        st.metric("Furnace Mean Temp (°C)", f"{furnace_results.get('mean_temp',0):.0f}")
        st.metric("Furnace PF", f"{furnace_results.get('mean_pf',0):.2f}")
        if electrode_results.get("paste_consumption"):
            st.metric("Electrode Paste (kg/ton)", f"{electrode_results['paste_consumption'].get('specific_consumption',0):.2f}")
        if process_results.get("equipment_utilization"):
            st.metric("Equipment Utilization (%)", f"{100*process_results['equipment_utilization'].get('overall_utilization',0):.2f}")
        st.subheader("Visualization")
        # Add furnace/process plots as needed

    with tab4:
        st.header("🔧 Furnace Optimization")
        if furnace_results.get("recommendations"):
            for rec in furnace_results["recommendations"]:
                st.success(rec)
        else:
            st.info("No furnace optimization recommendations available.")
        st.subheader("Visualization")
        # Add furnace optimization visualizations as needed

    with tab5:
        st.header("🔮 What-if Analysis")
        st.write("Scenarios:")
        for s in scenarios:
            st.info(f"Scenario: {s.get('change','')} - Impact: {s.get('impact','')}")
        st.write("Results:")
        st.dataframe(pd.DataFrame(whatif_results))
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
        st.metric("Potential Savings (USD)", f"${savings_results.get('estimated_savings_usd', 0):,.2f}")
        # Add cumulative savings chart, if desired

    with tab8:
        st.header("✔️ Data Validation")
        st.write(validation_results)

    with tab9:
        st.header("📊 Benchmarking")
        st.write(benchmarking_results)
        st.header("📝 Action Tracker")
        st.write(actions)
        if st.button("Add Action"):
            action_text = st.text_input("Enter new action:")
            if action_text:
                action_tracker.ActionTracker().add_action(action_text)
                st.success("Action added!")

    st.success("Analysis complete! Explore the tabs above for results.")

else:
    st.info("Upload all required CSVs or select random demo data, then click **Run Analysis**.")
