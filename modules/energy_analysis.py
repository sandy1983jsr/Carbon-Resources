import numpy as np
import pandas as pd

class EnergyAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def analyze(self):
        ec = self.datasets["energy_consumption"]
        res = {}
        res["avg_pf"] = ec["power_factor"].mean() if "power_factor" in ec.columns else None
        res["avg_intensity"] = ec["kwh_consumed"].mean()
        res["hourly"] = ec.groupby(ec["timestamp"].dt.hour)["kwh_consumed"].mean()
        res["area"] = (
            ec.groupby("process_area")["kwh_consumed"].sum() if "process_area" in ec.columns else None
        )
        # Anomaly detection - z score
        z = (ec["kwh_consumed"] - ec["kwh_consumed"].mean()) / (
            ec["kwh_consumed"].std() + 1e-6
        )
        res["anomaly_idx"] = np.where(abs(z) > 2)[0]
        res["anomaly_vals"] = ec.iloc[res["anomaly_idx"]][["timestamp", "kwh_consumed"]]
        # For trend
        ec["date"] = ec["timestamp"].dt.date
        res["daily"] = ec.groupby("date")["kwh_consumed"].sum()
        self.results = res
        return res
