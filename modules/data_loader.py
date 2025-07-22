import os
import pandas as pd
import yaml

class DataLoader:
    def __init__(self, data_dir, config_path):
        self.data_dir = data_dir
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.expected_files = self.config.get('data_files', {})

    def load_all_datasets(self):
        datasets = {}
        for dset, meta in self.expected_files.items():
            file_path = os.path.join(self.data_dir, meta['filename'])
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=meta.get('date_columns', []))
                datasets[dset] = df
        return datasets
