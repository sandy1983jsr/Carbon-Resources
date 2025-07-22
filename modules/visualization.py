import os
class DashboardGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_dashboard(self, output_dir):
        html = "<h2>Ferro Alloy Plant Dashboard</h2>"
        for k, results in self.kwargs.items():
            html += f"<h3>{k.replace('_', ' ').title()}</h3><ul>"
            if isinstance(results, dict):
                for key, val in results.items():
                    html += f"<li>{key}: {val}</li>"
            elif isinstance(results, list):
                for i, item in enumerate(results):
                    html += f"<li>Scenario {i+1}: {item.get('name', '')} - {item.get('description', '')}</li>"
            html += "</ul>"
        os.makedirs(f"{output_dir}/dashboard", exist_ok=True)
        with open(f"{output_dir}/dashboard/index.html", "w") as f:
            f.write(html)
