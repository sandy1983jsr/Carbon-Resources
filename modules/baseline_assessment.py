class BaselineAssessment:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def run_assessment(self):
        self.results['energy_total'] = self.datasets['energy_consumption']['kwh_consumed'].sum()
        input_tons = self.datasets['material_input']['quantity_tons'].sum()
        output_tons = self.datasets['material_output']['quantity_tons'].sum()
        self.results['material_yield'] = output_tons / input_tons if input_tons else None
        return self.results
