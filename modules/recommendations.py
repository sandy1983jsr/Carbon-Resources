def recommend_energy_saving(avg_pf, furnace_pf, energy_gap, yield_gap, paste_gap):
    actions = []
    # Power factor
    if avg_pf is not None and avg_pf < 0.95:
        actions.append({
            "title": "Correct Power Factor",
            "desc": "Install or upgrade PF correction capacitors to reduce kWh/ton.",
            "potential_savings_pct": abs(energy_gap) if energy_gap and energy_gap > 0 else 2.0,
            "type": "Quick win" if abs(energy_gap) < 3 else "Capex"
        })
    # Furnace
    if furnace_pf is not None and furnace_pf < 0.96:
        actions.append({
            "title": "Furnace PF Optimization",
            "desc": "Review transformer tap and load balancing.",
            "potential_savings_pct": 1.5,
            "type": "Quick win"
        })
    # Yield
    if yield_gap is not None and yield_gap < 0:
        actions.append({
            "title": "Improve Material Yield",
            "desc": "Tighten batching, minimize spillage, review material handling.",
            "potential_savings_pct": abs(yield_gap) * 100,
            "type": "Quick win" if abs(yield_gap) < 2 else "Capex"
        })
    # Electrode paste
    if paste_gap is not None and paste_gap > 0:
        actions.append({
            "title": "Reduce Electrode Paste Consumption",
            "desc": "Audit paste addition process and check paste quality/rheology.",
            "potential_savings_pct": abs(paste_gap),
            "type": "Quick win" if abs(paste_gap) < 2 else "Capex"
        })
    return actions
