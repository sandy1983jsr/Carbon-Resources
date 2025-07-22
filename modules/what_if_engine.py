class WhatIfEngine:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def generate_scenarios(self):
        return [
            {
                "name": "Reduce Furnace Setpoint by 20°C",
                "description": "Lower the main furnace temperature setpoint by 20°C to reduce specific energy consumption.",
                "category": "energy",
                "energy_savings_percentage": 3.8,
                "material_savings_percentage": 0.0,
                "cost_savings_percentage": 2.2,
                "implementation_difficulty": "medium",
                "investment_level": "low",
                "payback_period_months": 10,
                "actions": ["Review process control; calibrate temperature sensors"]
            },
            {
                "name": "Increase Mn-ore to Slag Ratio by 10%",
                "description": "Increase the proportion of Mn-ore relative to FeMn slag in the mix.",
                "category": "material",
                "energy_savings_percentage": 2.0,
                "material_savings_percentage": 1.5,
                "cost_savings_percentage": 1.5,
                "implementation_difficulty": "medium",
                "investment_level": "low",
                "payback_period_months": 8,
                "actions": ["Revise burden mix recipe; monitor slag/metal ratio"]
            },
            {
                "name": "Switch to Higher-Purity Manganese Ore",
                "description": "Adopt a higher-purity manganese ore to improve furnace efficiency.",
                "category": "material",
                "energy_savings_percentage": 4.2,
                "material_savings_percentage": 2.0,
                "cost_savings_percentage": 3.0,
                "implementation_difficulty": "medium",
                "investment_level": "high",
                "payback_period_months": 16,
                "actions": ["Negotiate with suppliers; adjust procurement contracts"]
            },
            {
                "name": "Maintain Power Factor at 0.96",
                "description": "Consistently operate with a power factor at or above 0.96.",
                "category": "energy",
                "energy_savings_percentage": 1.8,
                "material_savings_percentage": 0.0,
                "cost_savings_percentage": 1.2,
                "implementation_difficulty": "low",
                "investment_level": "medium",
                "payback_period_months": 12,
                "actions": ["Upgrade PF correction system; operator training"]
            },
            {
                "name": "Reduce Electrode Paste Consumption by 10%",
                "description": "Optimize electrode management and maintenance to lower paste usage.",
                "category": "material",
                "energy_savings_percentage": 0.0,
                "material_savings_percentage": 0.0,
                "cost_savings_percentage": 2.5,
                "implementation_difficulty": "medium",
                "investment_level": "low",
                "payback_period_months": 7,
                "actions": ["Implement predictive maintenance; tighter process control"]
            },
            {
                "name": "Increase Batch Weight by 5%",
                "description": "Increase batch weight to improve throughput and process stability.",
                "category": "process",
                "energy_savings_percentage": 1.5,
                "material_savings_percentage": 0.5,
                "cost_savings_percentage": 1.0,
                "implementation_difficulty": "medium",
                "investment_level": "low",
                "payback_period_months": 9,
                "actions": ["Review batch recipes; calibrate weighing systems"]
            },
            {
                "name": "Extend Shift Length from 8 to 10 Hours",
                "description": "Operate longer shifts to increase uptime and reduce changeover losses.",
                "category": "process",
                "energy_savings_percentage": 0.0,
                "material_savings_percentage": 0.0,
                "cost_savings_percentage": 1.0,
                "implementation_difficulty": "high",
                "investment_level": "low",
                "payback_period_months": 14,
                "actions": ["Negotiate with labor; update scheduling system"]
            },
            {
                "name": "Introduce Preheating for Raw Materials",
                "description": "Install a preheating system for raw materials to reduce furnace load.",
                "category": "energy",
                "energy_savings_percentage": 6.0,
                "material_savings_percentage": 0.0,
                "cost_savings_percentage": 4.5,
                "implementation_difficulty": "high",
                "investment_level": "high",
                "payback_period_months": 24,
                "actions": ["CAPEX for preheating equipment; process integration"]
            },
            {
                "name": "Recycle 20% More FeMn Slag",
                "description": "Increase recycling of FeMn slag to reduce waste and energy.",
                "category": "material",
                "energy_savings_percentage": 1.2,
                "material_savings_percentage": 0.8,
                "cost_savings_percentage": 0.8,
                "implementation_difficulty": "medium",
                "investment_level": "medium",
                "payback_period_months": 13,
                "actions": ["Upgrade slag handling system; process trials"]
            },
            {
                "name": "Integrate 20% Renewable Energy",
                "description": "Adopt solar or wind to meet part of plant's power demand.",
                "category": "energy",
                "energy_savings_percentage": 0.0,
                "material_savings_percentage": 0.0,
                "cost_savings_percentage": 1.8,
                "implementation_difficulty": "high",
                "investment_level": "very high",
                "payback_period_months": 36,
                "actions": ["Invest in renewable generation; grid integration"]
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
            },
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
            }
        ]

    def run_scenarios(self, scenarios):
        self.results = {"scenarios": scenarios}
        return self.results
