class ElectrodeOptimization:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def optimize(self):
        df = self.datasets.get("furnace_data")
        results = {}
        if (
            df is not None
            and "electrode_paste_consumption_kg" in df.columns
            and "production_tons" in df.columns
        ):
            total_paste = df["electrode_paste_consumption_kg"].sum()
            total_prod = df["production_tons"].sum()
            results["paste_consumption"] = {
                "total_paste_kg": total_paste,
                "specific_consumption": total_paste / total_prod if total_prod else None,
            }
        self.results = results
        return results
