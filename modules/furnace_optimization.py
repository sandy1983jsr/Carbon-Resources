class FurnaceOptimization:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def optimize(self):
        df = self.datasets.get("furnace_data")
        results = {}
        if df is not None:
            results["mean_temp"] = df["temperature"].mean() if "temperature" in df.columns else None
            results["mean_pf"] = df["power_factor"].mean() if "power_factor" in df.columns else None
            results["std_temp"] = df["temperature"].std() if "temperature" in df.columns else None
        self.results = results
        return results
