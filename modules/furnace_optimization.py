class FurnaceOptimization:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def optimize(self):
        df = self.datasets.get('furnace_data')
        results = {}
        if df is not None:
            results['mean_temp'] = df['temperature'].mean()
            results['mean_pf'] = df['power_factor'].mean()
            results['std_temp'] = df['temperature'].std()
        self.results = results
        return results
