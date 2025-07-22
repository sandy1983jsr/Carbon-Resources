import pandas as pd

class BaselineAssessment:
    """
    Performs a baseline assessment of energy, material, and production data.
    Returns total energy, material yield, and material loss as key baseline KPIs.
    """

    def __init__(self, datasets: dict):
        """
        :param datasets: dict of {str: pd.DataFrame}, as from DataLoader
        """
        self.datasets = datasets
        self.results = {}

    def run_assessment(self):
        res = {}

        # Baseline: Total Energy Consumption
        ec = self.datasets.get("energy_consumption")
        if ec is not None and "kwh_consumed" in ec.columns:
            res["energy_total"] = ec["kwh_consumed"].sum()
        else:
            res["energy_total"] = None

        # Baseline: Total Production (tons)
        prod = self.datasets.get("production")
        if prod is not None and "production_tons" in prod.columns:
            res["production_total"] = prod["production_tons"].sum()
        else:
            res["production_total"] = None

        # Baseline: Material Input & Output (tons)
        mi = self.datasets.get("material_input")
        mo = self.datasets.get("material_output")
        if mi is not None and "quantity_tons" in mi.columns:
            total_input = mi["quantity_tons"].sum()
        else:
            total_input = None

        if mo is not None and "quantity_tons" in mo.columns:
            total_output = mo["quantity_tons"].sum()
        else:
            total_output = None

        res["material_input_total"] = total_input
        res["material_output_total"] = total_output

        # Baseline: Material Yield and Loss
        if total_input and total_output is not None and total_input != 0:
            res["material_yield"] = total_output / total_input
            res["material_loss"] = total_input - total_output
            res["material_loss_pct"] = 100 * (1 - (total_output / total_input))
        else:
            res["material_yield"] = None
            res["material_loss"] = None
            res["material_loss_pct"] = None

        # Baseline: Energy Intensity (kWh/ton)
        if res.get("energy_total") and res.get("production_total") and res["production_total"] != 0:
            res["energy_intensity_kwh_per_ton"] = res["energy_total"] / res["production_total"]
        else:
            res["energy_intensity_kwh_per_ton"] = None

        self.results = res
        return res

    def to_dataframe(self):
        """
        Returns a single-row DataFrame of the baseline KPIs (for display or export).
        """
        if not self.results:
            self.run_assessment()
        return pd.DataFrame([self.results])
