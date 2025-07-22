class MaterialAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def analyze(self):
        inp = self.datasets.get('material_input')
        out = self.datasets.get('material_output')
        results = {}
        if inp is not None and out is not None:
            results['input_by_type'] = inp.groupby('material_type')['quantity_tons'].sum().to_dict()
            results['output_total'] = out['quantity_tons'].sum()
            total_input = inp['quantity_tons'].sum()
            total_output = out['quantity_tons'].sum()
            results['material_yield'] = total_output / total_input if total_input else None
            results['material_loss_percentage'] = 100 * (1 - results['material_yield']) if total_input else None
        self.results = results
        return results
