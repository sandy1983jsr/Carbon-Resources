import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_random_datasets(n_hours=24*7, freq='H'):
    """
    Generate realistic random datasets for 7 days (default) with hourly resolution.
    Returns a dict of DataFrames for the main data files expected by the app.
    """
    np.random.seed(42)
    base_time = datetime.now() - timedelta(hours=n_hours)
    timestamps = pd.date_range(base_time, periods=n_hours, freq=freq)
    
    # Simulate 3 areas/processes for energy and furnaces
    areas = ['Furnace1', 'Furnace2', 'Furnace3']
    material_types = ['Ore', 'Coke', 'Quartzite', 'Other']
    output_types = ['FeSi', 'Slag', 'Dust']

    # Energy Consumption: each area has a distinct pattern
    energy_consumption = []
    for area in areas:
        base = np.random.uniform(900, 1100)
        daily_cycle = 100 * np.sin(np.linspace(0, 6*np.pi, n_hours))  # 3 days cycle
        random = np.random.normal(0, 60, n_hours)
        pf = np.clip(np.random.normal(0.90, 0.03, n_hours), 0.85, 1.0)
        for i, t in enumerate(timestamps):
            energy_consumption.append({
                'timestamp': t,
                'area': area,
                'kWh': max(0, base + daily_cycle[i] + random[i]),
                'power_factor': pf[i],
            })
    df_energy = pd.DataFrame(energy_consumption)

    # Production: simulate 2 shifts per day, some downtime
    prod = []
    for t in timestamps:
        shift = 1 if (t.hour >= 6 and t.hour < 18) else 2
        is_running = (t.hour % 30 != 0)
        amount = np.random.uniform(15, 25) if is_running else 0
        prod.append({
            'timestamp': t,
            'shift': shift,
            'tons_produced': amount,
            'downtime': 0 if is_running else np.random.uniform(0.5, 3.0)
        })
    df_prod = pd.DataFrame(prod)

    # Material Input: simulate batch feeding, randomize input types
    mat_in = []
    for t in timestamps:
        for mt in material_types:
            base = {'Ore': 12, 'Coke': 7, 'Quartzite': 3, 'Other': 1}[mt]
            fluct = np.random.uniform(-2.0, 2.0)
            mat_in.append({
                'timestamp': t,
                'material_type': mt,
                'tons': max(0, base + fluct + np.random.normal(0, 0.6))
            })
    df_matin = pd.DataFrame(mat_in)

    # Material Output: simulate main product and byproducts
    mat_out = []
    for t in timestamps:
        for ot in output_types:
            if ot == 'FeSi':
                tons = np.random.uniform(18, 22)
            elif ot == 'Slag':
                tons = np.random.uniform(4, 7)
            else:
                tons = np.random.uniform(0.2, 0.6)
            mat_out.append({
                'timestamp': t,
                'output_type': ot,
                'tons': tons
            })
    df_matout = pd.DataFrame(mat_out)

    # Furnace Data: simulate T, V, I, PF for each furnace
    furnace_data = []
    for area in areas:
        for i, t in enumerate(timestamps):
            temp = np.random.normal(1500, 50)
            voltage = np.random.normal(100, 10)
            current = np.random.normal(38, 4)
            pf = np.clip(np.random.normal(0.88, 0.04), 0.82, 0.98)
            furnace_data.append({
                'timestamp': t,
                'furnace': area,
                'temperature_C': temp + 20 * np.sin(2 * np.pi * (i % 24) / 24),
                'voltage': voltage,
                'current': current,
                'power_factor': pf,
            })
    df_furnace = pd.DataFrame(furnace_data)

    # Process Data: simulate equipment utilization, paste addition, etc.
    process_data = []
    for t in timestamps:
        equip_util = np.clip(np.random.normal(0.93, 0.02), 0.85, 1.0)
        paste_kg = np.random.uniform(9, 12)
        process_data.append({
            'timestamp': t,
            'equipment_utilization': equip_util,
            'paste_added_kg': paste_kg,
            'remarks': np.random.choice(['', '', '', '', 'Scheduled Maint.', 'Short Break', 'Power Dip'])
        })
    df_process = pd.DataFrame(process_data)

    # Add some event logs for action tracker (if used)
    action_events = []
    for i in range(15):
        t = timestamps[np.random.randint(0, len(timestamps))]
        action_events.append({
            'timestamp': t,
            'action': np.random.choice(['Routine Check', 'Electrode Change', 'Slag Tap', 'Temp Adjustment', 'Parameter Review']),
            'status': np.random.choice(['Completed', 'Pending', 'In Progress']),
            'notes': np.random.choice(['', 'All OK', 'Follow-up needed', 'Delayed due to supply'])
        })
    df_actions = pd.DataFrame(action_events)

    # Optionally, generate some benchmarking reference data (e.g. last year's stats)
    df_benchmark = pd.DataFrame({
        'metric': ['energy_intensity', 'material_yield', 'equip_util', 'paste_consumption'],
        'benchmark_value': [12000, 0.87, 0.95, 10.5],
        'unit': ['kWh/ton', '', '', 'kg/ton']
    })

    # --- Add sample furnace optimization data here ---
    # Simulate before/after temperature profile for optimization visualization
    # Extended to 72 points (3 days, hourly)
    n_opt_points = 72
    now = datetime.now()
    opt_timestamps = [now - timedelta(hours=i) for i in reversed(range(n_opt_points))]
    temp_before = np.random.normal(1520, 20, n_opt_points)
    temp_after = temp_before - np.random.uniform(10, 25, n_opt_points)

    furnace_optimization = {
        "mean_temp": float(np.mean(temp_after)),
        "mean_pf": float(np.mean(np.random.uniform(0.88, 0.94, n_opt_points))),
        "recommendations": [
            "Reduce setpoint temperature by 15°C for improved efficiency.",
            "Monitor electrode consumption post-optimization."
        ],
        "temperature_profile": {
            "timestamp": [t.strftime("%Y-%m-%d %H:%M") for t in opt_timestamps],
            "before": temp_before.tolist(),
            "after": temp_after.tolist(),
        }
    }

    return {
        'energy_consumption': df_energy,
        'production': df_prod,
        'material_input': df_matin,
        'material_output': df_matout,
        'furnace_data': df_furnace,
        'process_data': df_process,
        'action_events': df_actions,
        'benchmark_data': df_benchmark,
        'furnace_optimization': furnace_optimization  # Unchanged, just more data points
    }
