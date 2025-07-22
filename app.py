import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import yaml
from modules.data_loader import DataLoader
from modules.baseline_assessment import BaselineAssessment
from modules.energy_analysis import EnergyAnalysis
from modules.material_analysis import MaterialAnalysis
from modules.furnace_optimization import FurnaceOptimization
from modules.electrode_optimization import ElectrodeOptimization
from modules.process_integration import ProcessIntegration
from modules.what_if_engine import WhatIfEngine
import modules.visualization as viz
import modules.validation as validation
import modules.benchmarking as benchmarking
import modules.recommendations as recommendations
import modules.action_tracker as action_tracker
import modules.savings_tracker as savings_tracker
import modules.sample_data as sample_data

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
        # Sample data download
        st.download_button(
            label=f"Download sample {label} file",
            data=sample_data.get_sample_csv(fname),
            file_name=fname,
            mime="text/csv",
        )
    config_file = st.file_uploader("Config file (optional)", type="yaml")
    st.markdown("---")
    st.info("Tip: For best results, upload all six files. Download templates above.")
    st.markdown("---")

    # Show first 5 rows of each csv after upload
    for fname, file in uploaded_files.items():
        if file:
            st.markdown(f"**Preview: {fname}**")
            df = pd.read_csv(file)
            st.dataframe(df.head())
            file.seek(0)  # Reset pointer for actual loading

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

    # Validation
    validation_results = validation.validate_all(datasets)
    for dset, missing in validation_results.items():
        if missing:
            st.warning(f"Missing columns in {dset}: {missing}")

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

    # --- Calculate Benchmarks and Recommendations ---
    # Example: energy intensity (kWh/ton)
    prod = datasets.get("production")
    if baseline_results.get("energy_total") and prod is not None and "production_tons" in prod.columns:
        actual_kwh_per_ton = baseline_results["energy_total"] / prod["production_tons"].sum()
    else:
        actual_kwh_per_ton = None
    energy_gap, energy_gap_pct = benchmarking.compare_to_benchmark("energy_total_kwh_per_ton", actual_kwh_per_ton)
    yield_gap, yield_gap_pct = benchmarking.compare_to_benchmark("material_yield", material_results.get("material_yield"))
    paste_gap, paste_gap_pct = benchmarking.compare_to_benchmark(
        "electrode_paste_kg_per_ton",
        electrode_results.get("paste_consumption", {}).get("specific_consumption"),
    )
    furnace_pf_gap, furnace_pf_gap_pct = benchmarking.compare_to_benchmark("furnace_pf", furnace_results.get("mean_pf"))
    rec_actions = recommendations.recommend_energy_saving(
        energy_results.get("avg_pf"), furnace_results.get("mean_pf"), energy_gap, yield_gap, paste_gap
    )

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Energy Dashboard", "Material Dashboard", "Process & Furnace", "Scenario/ROI", "Executive Summary", "Actions & Savings"
    ])

    with tab1:
        st.header("⚡ Energy Dashboard")
        st.metric("Total Energy Consumption (kWh)", f"{baseline_results.get('energy_total',0):,.0f}")
        st.metric("Average Power Factor", f"{energy_results.get('avg_pf',0):.2f}")
        st.metric("Average Energy Intensity (kWh/ton)", f"{actual_kwh_per_ton:.2f}" if actual_kwh_per_ton else "N/A")
        st.metric("Gap to Benchmark (kWh/ton)", f"{energy_gap:.2f}" if energy_gap else "N/A")
        st.markdown("**Energy by Area:**")
        area_data = energy_results.get("area")
        if area_data is not None:
            fig = viz.plot_energy_area_bar(area_data)
            if fig is not None:
                st.plotly_chart(fig)
        st.markdown("**Hourly Trend:**")
        hourly = energy_results.get("hourly")
        if hourly is not None:
            fig = viz.plot_energy_hourly(hourly)
            if fig is not None:
                st.plotly_chart(fig)
        st.markdown("**Daily Trend:**")
        daily = energy_results.get("daily")
        if daily is not None:
            fig = viz.plot_energy_trend(daily, anomalies=energy_results.get("anomaly_vals"))
            if fig is not None:
                st.plotly_chart(fig)
        st.info("Tip: Spikes indicate anomalies or process changes. Use anomaly highlights for root cause investigation.")

    with tab2:
        st.header("🧱 Material Dashboard")
        st.metric("Material Yield (%)", f"{100*(material_results.get('material_yield',0)):.2f}")
        st.metric("Material Loss (%)", f"{material_results.get('material_loss_percentage',0):.2f}")
        st.metric("Yield Gap to Benchmark", f"{yield_gap_pct:.2f}%" if yield_gap_pct else "N/A")
        input_by_type = material_results.get("input_by_type")
        st.markdown("**Input Material Split:**")
        if input_by_type:
            fig = viz.plot_material_pie(input_by_type)
            if fig is not None:
                st.plotly_chart(fig)
        st.markdown("**Material Yield Gauge:**")
        if material_results.get("material_yield") is not None and material_results.get("material_loss_percentage") is not None:
            fig = viz.plot_material_yield(material_results["material_yield"], material_results["material_loss_percentage"])
            if fig is not None:
                st.plotly_chart(fig)
        st.markdown("**Material Flow Sankey:**")
        st.plotly_chart(viz.plot_sankey())
        st.info("Wide 'waste' flows or low yield indicate batching and handling improvement opportunities.")

    with tab3:
        st.header("🔥 Process & Furnace Dashboard")
        st.metric("Furnace Mean Temp (°C)", f"{furnace_results.get('mean_temp',0):.0f}")
        st.metric("Furnace PF", f"{furnace_results.get('mean_pf',0):.2f}")
        st.metric("Furnace PF Gap", f"{furnace_pf_gap_pct:.2f}%" if furnace_pf_gap_pct else "N/A")
        if electrode_results.get("paste_consumption"):
            st.metric("Electrode Paste (kg/ton)", f"{electrode_results['paste_consumption'].get('specific_consumption',0):.2f}")
            st.metric("Paste Gap to Benchmark", f"{paste_gap_pct:.2f}%" if paste_gap_pct else "N/A")
        if process_results.get("equipment_utilization"):
            st.metric("Equipment Utilization (%)", f"{100*process_results['equipment_utilization'].get('overall_utilization',0):.2f}")
        st.info("Process KPIs highlight operational health and loss points.")

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
        st.info("Adjust parameters (PF, yield, etc.) in code for real-time what-if analysis.")

    with tab5:
        st.header("📊 Executive Summary")
        st.subheader("Top KPIs")
        st.write(f"- Energy Intensity: {actual_kwh_per_ton:.2f} kWh/ton (Gap: {energy_gap:.2f})")
        st.write(f"- Material Yield: {material_results.get('material_yield',0)*100:.2f}% (Gap: {yield_gap_pct:.2f}%)")
        st.write(f"- Furnace Power Factor: {furnace_results.get('mean_pf',0):.2f} (Gap: {furnace_pf_gap_pct:.2f}%)")
        st.write(f"- Electrode Paste: {electrode_results.get('paste_consumption',{}).get('specific_consumption',0):.2f} kg/ton (Gap: {paste_gap_pct:.2f}%)")
        st.subheader("Biggest Opportunities")
        for rec in rec_actions[:3]:
            st.write(f"- {rec['title']}: {rec['desc']} (Potential savings: {rec['potential_savings_pct']}%)")
        st.subheader("Recommended Next Actions")
        for rec in rec_actions:
            st.write(f"- {rec['title']} ({rec['type']}): {rec['desc']}")
        st.info("Executive summary highlights biggest gaps and top interventions for quick decision-making.")

    with tab6:
        st.header("📅 Actions & Savings Tracker")
        if "action_tracker" not in st.session_state:
            st.session_state["action_tracker"] = action_tracker.initialize_tracker()
        tracker_df = st.session_state["action_tracker"]
        st.dataframe(tracker_df)
        with st.form("add_action_form"):
            action = st.text_input("Action")
            owner = st.text_input("Owner")
            due = st.date_input("Due Date")
            status = st.selectbox("Status", ["Planned", "In Progress", "Done"])
            savings = st.number_input("Estimated Savings (₹ lakh)", min_value=0.0, value=0.0)
            completed = st.date_input("Completed On", value=pd.to_datetime("today"))
            submitted = st.form_submit_button("Add/Update Action")
            if submitted and action:
                st.session_state["action_tracker"] = action_tracker.add_action(
                    st.session_state["action_tracker"], action, owner, due, status, savings, completed
                )
        # Savings tracker
        savings_df = savings_tracker.update_savings_tracker(st.session_state["action_tracker"])
        st.subheader("Cumulative Savings")
        fig = viz.plot_savings_over_time(savings_df)
        if fig is not None:
            st.plotly_chart(fig)
        st.info("Track actions and realized savings to drive continuous improvement.")

    st.success("Consulting dashboard complete. Use each tab to guide client discussion and action planning.")

else:
    st.info("Upload all required CSVs and click **Run Analysis**.")
