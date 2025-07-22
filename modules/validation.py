import pandas as pd

REQUIRED_COLUMNS = {
    'energy_consumption': ['timestamp', 'kwh_consumed', 'process_area', 'power_factor'],
    'production': ['timestamp', 'production_tons', 'product_type'],
    'material_input': ['timestamp', 'material_type', 'quantity_tons'],
    'material_output': ['timestamp', 'product_type', 'quantity_tons'],
    'furnace_data': ['timestamp', 'temperature', 'power_factor', 'electrode_paste_consumption_kg', 'energy_consumed', 'production_tons'],
    'process_data': ['timestamp', 'process_area', 'uptime_hours', 'total_hours', 'batch_weight_actual', 'batch_weight_target'],
}

def validate_csv(df, expected_columns):
    missing = [col for col in expected_columns if col not in df.columns]
    return missing

def validate_all(datasets):
    results = {}
    for dset, columns in REQUIRED_COLUMNS.items():
        df = datasets.get(dset)
        if df is not None:
            missing = validate_csv(df, columns)
            results[dset] = missing
        else:
            results[dset] = columns  # All missing if dataset missing
    return results
