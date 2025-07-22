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
    df = pd.DataFrame({"area": list(energy_area.keys()), "energy": list(energy_area.values())})
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
    df = pd.DataFrame({"hour": hourly.index, "energy": hourly.values})
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
    df = pd.DataFrame({"date": daily.index, "energy": daily.values})
    fig = px.line(
        df, x="date", y="energy",
        title="Daily Energy Consumption Trend",
        markers=True,
    )
    fig.update_traces(line_color=COLOR_PALETTE["orange"])
    # anomalies
    if anomalies is not None and len(anomalies) > 0:
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
    df = pd.DataFrame({"type": list(input_by_type.keys()), "tons": list(input_by_type.values())})
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
    # Example Sankey for consulting demo
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
