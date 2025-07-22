class EnergyAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def analyze(self):
        ec = self.datasets['energy_consumption']
        self.results['avg_pf'] = ec['power_factor'].mean()
        self.results['avg_intensity'] = ec['kwh_consumed'].mean()
        return self.results
