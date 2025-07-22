import pandas as pd
import numpy as np

def generate_random_datasets(n_hours=24):
    # Generate hourly timestamps
    times = pd.date_range("2025-07-01", periods=n_hours, freq="H")
    
    # Energy Consumption Data
    energy_consumption = pd.DataFrame({
        "timestamp": times,
        "kwh_consumed": np.random.normal(2200, 200, size=n_hours).round(),
        "process_area": np.random.choice(["Furnace", "Conveyor", "Electrode Paste"], size=n_hours),
        "power_factor": np.random.uniform(0.85, 0.96, size=n_hours).round(2),
    })

    # Production Data
    production = pd.DataFrame({
        "timestamp": times,
        "production_tons": np.random.normal(5.0, 0.4, size=n_hours).round(2),
        "product_type": "SiMn",
    })

    # Material Input Data
    material_input = pd.DataFrame({
        "timestamp": np.repeat(times[0], 3),
        "material_type": ["Manganese Ore", "Silica", "Coal"],
        "quantity_tons": np.random.uniform(1, 5, size=3).round(2),
    })

    # Material Output Data
    material_output = pd.DataFrame({
        "timestamp": times[:3],
        "product_type": "SiMn",
        "quantity_tons": np.random.normal(5.0, 0.2, size=3).round(2),
    })

    # Furnace Data
    furnace_data = pd.DataFrame({
        "timestamp": times,
        "temperature": np.random.normal(1650, 20, size=n_hours).round(),
        "power_factor": np.random.uniform(0.80, 0.96, size=n_hours).round(2),
        "electrode_paste_consumption_kg": np.random.normal(120, 5, size=n_hours).round(),
        "energy_consumed": np.random.normal(2200, 200, size=n_hours).round(),
        "production_tons": np.random.normal(5.0, 0.4, size=n_hours).round(2),
    })

    # Process Data
    process_data = pd.DataFrame({
        "timestamp": times,
        "process_area": np.random.choice(
            ["Raw Material Handling", "Furnace", "Conveyor"], size=n_hours
        ),
        "uptime_hours": np.random.uniform(0.9, 1.0, size=n_hours).round(2),
        "total_hours": 1.0,
        "batch_weight_actual": np.random.normal(5.75, 0.1, size=n_hours).round(2),
        "batch_weight_target": 5.7
    })

    # Estimated Savings Data (cumulative over time)
    estimated_savings = pd.DataFrame({
        "Date": times,
        "Cumulative Savings": np.cumsum(np.random.uniform(30, 120, n_hours)).round(2),
        "Comment": ["Energy optimization"] * n_hours
    })

    # Benchmarking Data
    benchmarking_data = pd.DataFrame({
        "metric": ["energy_intensity", "material_yield", "equip_util", "paste_consumption"],
        "benchmark_value": [11800, 0.91, 0.97, 11.3],
        "unit": ["kWh/ton", "", "", "kg/ton"]
    })

    # Action Tracker Data
    action_tracker = pd.DataFrame({
        "timestamp": np.random.choice(times, 10),
        "action": np.random.choice([
            "Routine Check", "Electrode Change", "Slag Tap", 
            "Temp Adjustment", "Parameter Review"], size=10),
        "status": np.random.choice([
            "Completed", "Pending", "In Progress"], size=10),
        "notes": np.random.choice([
            "", "All OK", "Follow-up needed", "Delayed due to supply"], size=10)
    }).sort_values("timestamp").reset_index(drop=True)

    # What-if Analysis Data
    what_if_analysis = pd.DataFrame([
        {
            "scenario": "Reduce Furnace Setpoint by 20°C",
            "expected_energy_savings_%": 3.8,
            "expected_cost_savings_%": 2.2,
            "impact": "Lower energy consumption, potential minor yield loss"
        },
        {
            "scenario": "Increase Shift Length by 2 hours",
            "expected_energy_savings_%": 0.0,
            "expected_cost_savings_%": 1.0,
            "impact": "More production per shift, minor maintenance risk"
        }
    ])

    # Recommendations Data
    recommendations = [
        "Optimize furnace temperature setpoint to reduce energy consumption.",
        "Regularly monitor electrode paste consumption for early anomaly detection.",
        "Improve process area insulation to minimize heat loss.",
        "Review batch weight targets for improved material yield."
    ]

    return {
        "energy_consumption": energy_consumption,
        "production": production,
        "material_input": material_input,
        "material_output": material_output,
        "furnace_data": furnace_data,
        "process_data": process_data,
        "estimated_savings": estimated_savings,
        "benchmark_data": benchmarking_data,
        "action_events": action_tracker,
        "what_if_analysis": what_if_analysis,
        "recommendations": recommendations,
    }
