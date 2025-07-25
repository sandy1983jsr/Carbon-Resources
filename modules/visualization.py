import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

COLOR_PALETTE = {
    "orange": "#FF9800",
    "grey": "#9E9E9E",
    "background": "#FFFFFF",
    "text": "#212121",
    "accent": "#424242",
}

def plot_energy_area_bar(energy_area):
    if energy_area is None or len(energy_area) == 0:
        return None
    if isinstance(energy_area, pd.Series):
        area_names = list(energy_area.index.astype(str))
        area_vals = list(energy_area.values)
    elif isinstance(energy_area, dict):
        area_names = list(energy_area.keys())
        area_vals = list(energy_area.values())
    else:
        try:
            area_names = list(energy_area.keys())
            area_vals = list(energy_area.values())
        except Exception:
            return None
    df = pd.DataFrame({"area": area_names, "energy": area_vals})
    fig = px.bar(
        df, x="area", y="energy",
        color="area",
        color_discrete_sequence=[COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["orange"]],
        title="Energy Consumption by Area"
    )
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
        showlegend=False,
    )
    return fig

def plot_energy_hourly(hourly):
    if hourly is None or len(hourly) == 0:
        return None
    if isinstance(hourly, pd.Series):
        hours = list(hourly.index)
        vals = list(hourly.values)
    elif isinstance(hourly, dict):
        hours = list(hourly.keys())
        vals = list(hourly.values())
    else:
        try:
            hours = list(hourly.keys())
            vals = list(hourly.values())
        except Exception:
            return None
    df = pd.DataFrame({"hour": hours, "energy": vals})
    fig = px.line(
        df, x="hour", y="energy",
        line_shape="spline",
        markers=True,
        title="Hourly Energy Consumption Trend",
    )
    fig.update_traces(line_color=COLOR_PALETTE["orange"])
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
    )
    return fig

def plot_energy_trend(daily, anomalies=None):
    if daily is None or len(daily) == 0:
        return None
    if isinstance(daily, pd.Series):
        dates = list(daily.index)
        vals = list(daily.values)
    elif isinstance(daily, dict):
        dates = list(daily.keys())
        vals = list(daily.values())
    else:
        try:
            dates = list(daily.index)
            vals = list(daily.values)
        except Exception:
            return None
    df = pd.DataFrame({"date": dates, "energy": vals})
    fig = px.line(
        df, x="date", y="energy",
        title="Daily Energy Consumption Trend",
        markers=True,
    )
    fig.update_traces(line_color=COLOR_PALETTE["orange"])
    if anomalies is not None and hasattr(anomalies, "timestamp"):
        fig.add_scatter(
            x=anomalies["timestamp"], y=anomalies["kwh_consumed"],
            mode="markers", marker=dict(color=COLOR_PALETTE["grey"], size=10, symbol="x"),
            name="Anomaly"
        )
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
    )
    return fig

def plot_material_pie(input_by_type):
    if input_by_type is None or len(input_by_type) == 0:
        return None
    if isinstance(input_by_type, pd.Series):
        names = list(input_by_type.index.astype(str))
        vals = list(input_by_type.values)
    elif isinstance(input_by_type, dict):
        names = list(input_by_type.keys())
        vals = list(input_by_type.values())
    else:
        try:
            names = list(input_by_type.keys())
            vals = list(input_by_type.values())
        except Exception:
            return None
    df = pd.DataFrame({"type": names, "tons": vals})
    fig = px.pie(
        df, names="type", values="tons",
        color_discrete_sequence=[COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["background"]],
        title="Material Input by Type"
    )
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
    )
    return fig

def plot_material_yield(yield_val, loss_pct):
    if yield_val is None or loss_pct is None:
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=yield_val * 100,
        delta={'reference': 100 - loss_pct, 'increasing': {'color': COLOR_PALETTE["orange"]}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': COLOR_PALETTE["orange"]},
            'steps': [
                {'range': [0, 90], 'color': COLOR_PALETTE["grey"]},
                {'range': [90, 100], 'color': COLOR_PALETTE["orange"]}
            ],
        },
        title={'text': "Material Yield (%)", 'font': {'color': COLOR_PALETTE["orange"]}},
    ))
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
    )
    return fig

def plot_sankey():
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color=COLOR_PALETTE["accent"], width=0.5),
            label=["Storage", "Furnace", "Electrode", "Output", "Waste"],
            color=[COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["background"]]
        ),
        link=dict(
            source=[0,1,1,2],
            target=[1,2,3,4],
            value=[100,70,15,15],
            color=[COLOR_PALETTE["orange"], COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["grey"]]
        )
    )])
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"]
    )
    return fig

def plot_scenario_roi(scenarios):
    if not scenarios or len(scenarios) == 0:
        return None
    names = [s["name"] for s in scenarios]
    cost_save = [s["cost_savings_percentage"] for s in scenarios]
    fig = px.bar(
        x=names, y=cost_save,
        color=names,
        color_discrete_sequence=[COLOR_PALETTE["orange"], COLOR_PALETTE["grey"], COLOR_PALETTE["orange"]],
        title="Scenario ROI: Cost Savings (%)",
        labels={"x": "Scenario", "y": "Cost Savings (%)"}
    )
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
        showlegend=False,
    )
    return fig

def plot_savings_over_time(savings_df):
    if savings_df is None or savings_df.empty:
        return None
    fig = px.line(savings_df, x="Date", y="Cumulative Savings",
                  title="Cumulative Savings Over Time",
                  markers=True)
    fig.update_traces(line_color=COLOR_PALETTE["orange"])
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
    )
    return fig

def plot_furnace_optimization(furnace_results):
    """
    Plots a before/after comparison for furnace optimization, e.g., temperature profile.
    Expects furnace_results["temperature_profile"] to be a dict with keys "timestamp", "before", "after".
    """
    if (not isinstance(furnace_results, dict) or
        "temperature_profile" not in furnace_results or
        furnace_results["temperature_profile"] is None):
        return None

    profile = furnace_results["temperature_profile"]
    if not all(k in profile for k in ("timestamp", "before", "after")):
        return None

    df = pd.DataFrame({
        "timestamp": profile["timestamp"],
        "Before Optimization": profile["before"],
        "After Optimization": profile["after"],
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["Before Optimization"],
        mode="lines+markers", name="Before Optimization", 
        line=dict(color=COLOR_PALETTE["grey"], dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["After Optimization"],
        mode="lines+markers", name="After Optimization", 
        line=dict(color=COLOR_PALETTE["orange"])
    ))

    fig.update_layout(
        title="Furnace Temperature Profile: Before vs After Optimization",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        plot_bgcolor=COLOR_PALETTE["background"],
        paper_bgcolor=COLOR_PALETTE["background"],
        font_color=COLOR_PALETTE["text"],
        title_font_color=COLOR_PALETTE["orange"],
    )
    return fig
