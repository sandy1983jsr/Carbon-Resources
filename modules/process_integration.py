class ProcessIntegration:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def optimize(self):
        df = self.datasets.get('process_data')
        results = {}
        if df is not None and 'uptime_hours' in df.columns and 'total_hours' in df.columns:
            util = df['uptime_hours'].sum() / df['total_hours'].sum() if df['total_hours'].sum() else None
            results['equipment_utilization'] = {'overall_utilization': util}
        self.results = results
        return results
