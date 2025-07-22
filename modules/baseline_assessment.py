class BaselineAssessment:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def run_assessment(self):
        res = {}
        # Energy
        if "energy_consumption" in self.datasets:
            res["energy_total"] = self.datasets["energy_consumption"]["kwh_consumed"].sum()
        # Material yield
        if "material_input" in self.datasets and "material_output" in self.datasets:
            input_tons = self.datasets["material_input"]["quantity_tons"].sum()
            output_tons = self.datasets["material_output"]["quantity_tons"].sum()
            res["material_yield"] = output_tons / input_tons if input_tons else None
            res["material_loss"] = input_tons - output_tons
        self.results = res
        return res
