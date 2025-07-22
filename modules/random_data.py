import pandas as pd
import numpy as np

def generate_random_datasets(n_hours=24):
    # Energy Consumption
    times = pd.date_range("2025-07-01", periods=n_hours, freq="H")
    energy_consumption = pd.DataFrame({
        "timestamp": times,
        "kwh_consumed": np.random.normal(2200, 200, size=n_hours).round(),
        "process_area": np.random.choice(["Furnace", "Conveyor", "Electrode Paste"], size=n_hours),
        "power_factor": np.random.uniform(0.85, 0.96, size=n_hours).round(2),
    })

    # Production
    production = pd.DataFrame({
        "timestamp": times,
        "production_tons": np.random.normal(5.0, 0.4, size=n_hours).round(2),
        "product_type": "SiMn",
    })

    # Material Input
    material_input = pd.DataFrame({
        "timestamp": np.repeat(times[0], 3),
        "material_type": ["Manganese Ore", "Silica", "Coal"],
        "quantity_tons": np.random.uniform(1, 5, size=3).round(2),
    })

    # Material Output
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
        "process_area": np.random.choice(["Raw Material Handling", "Furnace", "Conveyor"], size=n_hours),
        "uptime_hours": np.random.uniform(0.9, 1.0, size=n_hours).round(2),
        "total_hours": 1.0,
        "batch_weight_actual": np.random.normal(5.75, 0.1, size=n_hours).round(2),
        "batch_weight_target": 5.7
    })

    return {
        "energy_consumption": energy_consumption,
        "production": production,
        "material_input": material_input,
        "material_output": material_output,
        "furnace_data": furnace_data,
        "process_data": process_data,
    }
