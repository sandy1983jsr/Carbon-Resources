class BaselineAssessment:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def run_assessment(self):
        res = {}

        # Energy consumption total
        if "energy_consumption" in self.datasets:
            ec = self.datasets["energy_consumption"]
            if "kwh_consumed" in ec.columns:
                res["energy_total"] = ec["kwh_consumed"].sum()
            else:
                res["energy_total"] = None
        else:
            res["energy_total"] = None

        # Material yield and loss
        if "material_input" in self.datasets and "material_output" in self.datasets:
            mi = self.datasets["material_input"]
            mo = self.datasets["material_output"]
            input_tons = mi["quantity_tons"].sum() if "quantity_tons" in mi.columns else None
            output_tons = mo["quantity_tons"].sum() if "quantity_tons" in mo.columns else None
            if input_tons and output_tons and input_tons != 0:
                res["material_yield"] = output_tons / input_tons
                res["material_loss"] = input_tons - output_tons
            else:
                res["material_yield"] = None
                res["material_loss"] = None
        else:
            res["material_yield"] = None
            res["material_loss"] = None

        self.results = res
        return res
