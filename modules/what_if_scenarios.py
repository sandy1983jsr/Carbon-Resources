import pandas as pd

def get_what_if_scenarios():
    return pd.DataFrame([
        {
            "scenario": "Reduce Furnace Setpoint by 20°C",
            "expected_energy_savings_%": 3.8,
            "expected_cost_savings_%": 2.2,
            "impact": "Lower energy use, potential minor yield loss"
        },
        {
            "scenario": "Increase Mn-ore to slag ratio by 10%",
            "expected_energy_savings_%": 2.0,
            "expected_cost_savings_%": 1.5,
            "impact": "Lower slag volume, reduced energy per ton"
        },
        {
            "scenario": "Switch to higher-purity manganese ore",
            "expected_energy_savings_%": 4.2,
            "expected_cost_savings_%": 3.0,
            "impact": "Improved yield and energy efficiency"
        },
        {
            "scenario": "Maintain power factor at 0.96",
            "expected_energy_savings_%": 1.8,
            "expected_cost_savings_%": 1.2,
            "impact": "Better electricity rates, less waste"
        },
        {
            "scenario": "Reduce electrode paste consumption by 10%",
            "expected_energy_savings_%": 0.0,
            "expected_cost_savings_%": 2.5,
            "impact": "Lower consumable cost, stable operation"
        },
        {
            "scenario": "Increase batch weight by 5%",
            "expected_energy_savings_%": 1.5,
            "expected_cost_savings_%": 1.0,
            "impact": "Higher throughput, more stable furnace"
        },
        {
            "scenario": "Extend shift length from 8 to 10 hours",
            "expected_energy_savings_%": 0.0,
            "expected_cost_savings_%": 1.0,
            "impact": "Improved uptime, potential staff fatigue"
        },
        {
            "scenario": "Introduce preheating for raw materials",
            "expected_energy_savings_%": 6.0,
            "expected_cost_savings_%": 4.5,
            "impact": "Lower furnace energy demand, some capital cost"
        },
        {
            "scenario": "Recycle 20% more FeMn slag",
            "expected_energy_savings_%": 1.2,
            "expected_cost_savings_%": 0.8,
            "impact": "Lower disposal, modest energy savings"
        },
        {
            "scenario": "Integrate 20% renewable energy",
            "expected_energy_savings_%": 0.0,
            "expected_cost_savings_%": 1.8,
            "impact": "Lower emissions, lower grid cost"
        },
    ])
