class MaterialAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def analyze(self):
        inp = self.datasets.get("material_input")
        out = self.datasets.get("material_output")
        results = {}
        if inp is not None and out is not None:
            if "material_type" in inp.columns and "quantity_tons" in inp.columns:
                results["input_by_type"] = inp.groupby("material_type")["quantity_tons"].sum().to_dict()
            else:
                results["input_by_type"] = None
            if "quantity_tons" in out.columns:
                results["output_total"] = out["quantity_tons"].sum()
            else:
                results["output_total"] = None
            total_input = inp["quantity_tons"].sum() if "quantity_tons" in inp.columns else None
            total_output = out["quantity_tons"].sum() if "quantity_tons" in out.columns else None
            if total_input and total_output and total_input != 0:
                results["material_yield"] = total_output / total_input
                results["material_loss_percentage"] = 100 * (1 - results["material_yield"])
            else:
                results["material_yield"] = None
                results["material_loss_percentage"] = None
        else:
            results["input_by_type"] = None
            results["output_total"] = None
            results["material_yield"] = None
            results["material_loss_percentage"] = None
        self.results = results
        return results
