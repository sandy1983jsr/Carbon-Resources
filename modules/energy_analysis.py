import numpy as np
import pandas as pd

class EnergyAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def analyze(self):
        if "energy_consumption" not in self.datasets:
            return {}
        ec = self.datasets["energy_consumption"]
        res = {}
        if "power_factor" in ec.columns:
            res["avg_pf"] = ec["power_factor"].mean()
        else:
            res["avg_pf"] = None
        if "kwh_consumed" in ec.columns:
            res["avg_intensity"] = ec["kwh_consumed"].mean()
        else:
            res["avg_intensity"] = None
        if "timestamp" in ec.columns and "kwh_consumed" in ec.columns:
            try:
                ec["timestamp"] = pd.to_datetime(ec["timestamp"])
                res["hourly"] = ec.groupby(ec["timestamp"].dt.hour)["kwh_consumed"].mean()
                res["daily"] = ec.groupby(ec["timestamp"].dt.date)["kwh_consumed"].sum()
            except Exception:
                res["hourly"] = None
                res["daily"] = None
        else:
            res["hourly"] = None
            res["daily"] = None
        if "process_area" in ec.columns and "kwh_consumed" in ec.columns:
            res["area"] = ec.groupby("process_area")["kwh_consumed"].sum()
        else:
            res["area"] = None
        # Anomaly detection - z score
        if "kwh_consumed" in ec.columns:
            z = (ec["kwh_consumed"] - ec["kwh_consumed"].mean()) / (ec["kwh_consumed"].std() + 1e-6)
            res["anomaly_idx"] = np.where(abs(z) > 2)[0]
            if "timestamp" in ec.columns:
                res["anomaly_vals"] = ec.iloc[res["anomaly_idx"]][["timestamp", "kwh_consumed"]]
            else:
                res["anomaly_vals"] = None
        else:
            res["anomaly_idx"] = []
            res["anomaly_vals"] = None
        self.results = res
        return res
