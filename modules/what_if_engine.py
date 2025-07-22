class WhatIfEngine:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def generate_scenarios(self):
        return [
            {
                "name": "Power Factor Correction",
                "description": "Improve PF to 0.95",
                "category": "energy",
                "energy_savings_percentage": 2.5,
                "material_savings_percentage": 0,
                "cost_savings_percentage": 2.0,
                "implementation_difficulty": "medium",
                "investment_level": "low",
                "payback_period_months": 12,
                "actions": ["Install PF correction capacitors"]
            },
            {
                "name": "Batch Weighing Improvement",
                "description": "Reduce over-batching losses with better weighing control.",
                "category": "material",
                "energy_savings_percentage": 0,
                "material_savings_percentage": 3.0,
                "cost_savings_percentage": 2.0,
                "implementation_difficulty": "low",
                "investment_level": "low",
                "payback_period_months": 6,
                "actions": ["Upgrade batch weighing system"]
            }
        ]

    def run_scenarios(self, scenarios):
        self.results = {"scenarios": scenarios}
