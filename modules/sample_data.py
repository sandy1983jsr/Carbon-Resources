import pandas as pd
from io import StringIO

SAMPLES = {
    "energy_consumption.csv": StringIO(
"""timestamp,kwh_consumed,process_area,power_factor
2025-07-01 00:00:00,2400,Furnace,0.91
2025-07-01 01:00:00,2300,Furnace,0.92
2025-07-01 02:00:00,2250,Furnace,0.93
2025-07-01 00:00:00,1350,Conveyor,0.89
2025-07-01 01:00:00,1400,Electrode Paste,0.85
"""
    ),
    "production.csv": StringIO(
"""timestamp,production_tons,product_type
2025-07-01 00:00:00,5.2,SiMn
2025-07-01 01:00:00,5.1,SiMn
2025-07-01 02:00:00,5.0,SiMn
"""
    ),
    "material_input.csv": StringIO(
"""timestamp,material_type,quantity_tons
2025-07-01 00:00:00,Manganese Ore,3.8
2025-07-01 00:00:00,Silica,1.2
2025-07-01 00:00:00,Coal,0.7
"""
    ),
    "material_output.csv": StringIO(
"""timestamp,product_type,quantity_tons
2025-07-01 00:00:00,SiMn,5.2
2025-07-01 01:00:00,SiMn,5.1
"""
    ),
    "furnace_data.csv": StringIO(
"""timestamp,temperature,power_factor,electrode_paste_consumption_kg,energy_consumed,production_tons
2025-07-01 00:00:00,1650,0.91,120,2400,5.2
2025-07-01 01:00:00,1645,0.92,118,2300,5.1
2025-07-01 02:00:00,1648,0.93,119,2250,5.0
"""
    ),
    "process_data.csv": StringIO(
"""timestamp,process_area,uptime_hours,total_hours,batch_weight_actual,batch_weight_target
2025-07-01 00:00:00,Raw Material Handling,0.95,1.0,5.75,5.7
2025-07-01 01:00:00,Furnace,0.99,1.0,5.78,5.7
2025-07-01 02:00:00,Conveyor,0.97,1.0,5.76,5.7
"""
    ),
}

def get_sample_csv(name):
    return SAMPLES[name].getvalue()
def get_sample_df(name):
    return pd.read_csv(SAMPLES[name])
def list_samples():
    return list(SAMPLES.keys())
