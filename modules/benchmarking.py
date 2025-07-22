# Simple hardcoded benchmarks for demo; could be loaded from YAML in production
BENCHMARKS = {
    'energy_total_kwh_per_ton': 3800,  # Best practice for ELC SiMn, example value
    'material_yield': 0.97,
    'electrode_paste_kg_per_ton': 22,
    'furnace_pf': 0.96,
}

def get_benchmark(metric):
    return BENCHMARKS.get(metric)

def compare_to_benchmark(metric, value):
    bench = get_benchmark(metric)
    if bench is None or value is None:
        return None, None
    gap = value - bench
    gap_pct = (gap / bench) * 100
    return gap, gap_pct
