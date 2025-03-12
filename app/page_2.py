from pandas.plotting import boxplot
import streamlit as st
import pandas as pd
import altair as alt
from figures_functions import *
import plotly.figure_factory as ff
import plotly.express as px
import datetime
import numpy as np
from db import (
    get_scenario_year_data,
    get_all_scenarios,
    generate_scenario_year_data,
    get_filter_nodes_for_gate,
    get_available_years,
    get_all_gate_settings,
)

from typing import Dict, List, Any
from dataclasses import dataclass


@st.cache_data
def load_scenario_data(scenario, year):
    scenario_year_data = get_scenario_year_data(scenario, year)
    scenario_data = generate_scenario_year_data(scenario_year_data)
    print(scenario_data.keys())
    return scenario_year_data, scenario_data

def full_post_process_hydro_data(
    data, model, year=None, start_date=None, end_date=None
):
    # print(data)
    hydro_df = data["water_levels"]
    if year:
        hydro_df["year"] = hydro_df["datetime"].dt.year
        hydro_df = hydro_df[hydro_df["year"] == year]
    hydro_df = hydro_df.loc[hydro_df["datetime"].dt.month.between(5, 11)]
    hydro_df["time_unit"] = 0.25
    hydro_df = hydro_df.rename(columns={"value": "water_level"})
    hydro_df["week"] = hydro_df["datetime"].dt.isocalendar().week
    # hydro_df = hydro_df[hydro_df.node == gate]
    return hydro_df

def get_scenario_and_year_selection(
    column,
    scenario_key,
    year_key,
    previous_scenario_key,
    available_years_key,
    previous_year_key,
):
    selected_model = st.selectbox(f"Select Scenario:", scenario_list, key=scenario_key)

    if selected_model != st.session_state[previous_scenario_key]:
        st.session_state[available_years_key] = get_available_years(selected_model)
        st.session_state[previous_scenario_key] = selected_model
        st.session_state[previous_year_key] = None

    if st.session_state[available_years_key]:
        selected_year = st.selectbox(
            f"Select Year:", st.session_state[available_years_key], key=year_key
        )
    else:
        selected_year = None

    return selected_model, selected_year


@dataclass
class ScenarioPlotConfig:
    key: str
    legend: None | Dict = None
    xaxis: None | Dict = None
    yaxis: None | Dict = None


def render_scenario(
    data: Dict[str, DataFrame],
    hydro_data,
    selected_model: str,
    selected_year: Any,
    month_names: Dict[int, str],
    boxplot_config: ScenarioPlotConfig,
    violin_config: ScenarioPlotConfig,
):
    ops_data = data["ops_data"].sort_values(by="datetime")
    ops_data["month_name"] = ops_data["datetime"].dt.month.map(month_names)

    st.success(f"Scenario Loaded: {selected_model} ({selected_year})")

    st.dataframe(
        ops_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": None,
            "scenario_id": None,
            "node": None,
            "gate_min_datetime": None,
            "gate_max_datetime": None,
            "year_x": st.column_config.NumberColumn(format="%d"),
        },
    )

    avg_daily_velocity_display = data["avg_daily_velocity"].copy()
    avg_daily_velocity_display["is_over_8fs"] = avg_daily_velocity_display[
        "is_over_8fs"
    ].map(
        {
            False: "Average Daily (hours) Under 8 ft/s",
            True: "Average Daily (hours) Over 8 ft/s",
        }  # type: Ignore
    )
    avg_daily_velocity_display = avg_daily_velocity_display.pivot_table(
        index="gate", columns="is_over_8fs", values="time_unit"
    ).reset_index()
    streak_duration_display = data["total_daily_velocity"].copy()
    streak_duration_display["is_over_8fs"] = streak_duration_display["is_over_8fs"].map(
        {
            False: "Average Streak (Hours) Under 8 ft/s",
            True: "Average Streak (Hours) Over 8 ft/s",
        }
    )
    streak_duration_display = streak_duration_display.pivot_table(
        index="gate",
        columns="is_over_8fs",
        values="daily_average_time_per_consecutive_group",
    ).reset_index()

    velocity_summary_data = pd.merge(
        avg_daily_velocity_display, streak_duration_display, on="gate"
    )

    st.dataframe(
        velocity_summary_data.style.highlight_max(
            axis=0,
            subset=[col for col in velocity_summary_data.columns if col != "gate"],
            props="background-color: #a2cf9d; color: black; font-weight: bold; border: 2px solid green;",
        ),
        hide_index=True,
        column_config={
            "gate": "Gate",
            "Average Daily (hours) Under 8 ft/s": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Average Daily (hours) Over 8 ft/s": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Average Streak (Hours) Under 8 ft/s": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Average Streak (Hours) Over 8 ft/s": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )

    avg_daily_gate_display = data["avg_daily_gate"].copy()
    avg_daily_gate_display["gate_status"] = avg_daily_gate_display["gate_status"].map(
        {
            "Closed": "Average Daily Closed (Hours)",
            "Open": "Average Daily Open (Hours)",
        }
    )
    avg_daily_gate_display = avg_daily_gate_display.pivot_table(
        index="gate", columns="gate_status", values="time_unit"
    ).reset_index()
    total_daily_gate_display = data["total_daily_gate"].copy()
    total_daily_gate_display["gate_status"] = total_daily_gate_display[
        "gate_status"
    ].map(
        {
            "Closed": "Average Closed Duration per Streak",
            "Open": "Average Open Duration per Streak",
        }
    )
    total_daily_gate_display = total_daily_gate_display.pivot_table(
        index="gate",
        columns="gate_status",
        values="daily_average_time_per_consecutive_gate",
    ).reset_index()

    gate_summary_data = pd.merge(
        avg_daily_gate_display, total_daily_gate_display, on="gate"
    )

    st.dataframe(
        gate_summary_data.style.highlight_max(
            axis=0,
            subset=[col for col in gate_summary_data.columns if col != "gate"],
            props="background-color: #a2cf9d; color: black; font-weight: bold; border: 2px solid green;",
        ),
        column_config={
            "gate": "Gate",
            "Average Daily Closed (Hours)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Average Daily Open (Hours)": st.column_config.NumberColumn(format="%.2f"),
            "Average Closed Duration per Streak": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Average Open Duration per Streak": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
        hide_index=True,
    )

    min_max_summary_display = data["ops_data"].copy()
    min_max_summary_display = (
        min_max_summary_display.groupby("gate")["velocity"]
        .agg(["min", "max"])
        .reset_index()
    )
    min_max_summary_display = min_max_summary_display.rename(
        columns={
            "min": "Minimum Velocity Through Gate (ft/s)",
            "max": "Maximum Velocity Through Gate (ft/s)",
        }
    )

    st.dataframe(
        min_max_summary_display,
        hide_index=True,
        column_config={
            "gate": "Gate",
            "Minimum Velocity Through Gate (ft/s)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Maximum Velocity Through Gate (ft/s)": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )

    if not ops_data.empty:
        boxplot = px.box(
            ops_data,
            x="month_name",
            y="velocity",
            color="gate",
            category_orders={"month_name": list(month_names.values())},
            title=f"Velocity by Month (May-Nov) - {selected_model} ({selected_year})",
            labels={
                "month_name": "Month",
                "value": "velocity",
                "node": "Location",
            },
            height=500,
            points="outliers",
        )

        boxplot.update_traces(
            hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>velocity: %{y:.3f} CFS<extra></extra>",
            customdata=[[node] for node in ops_data["gate"]],
        )

        boxplot.update_layout(
            boxmode="group",
            legend=dict(
                orientation="h",
                y=1.1,
                x=0.5,
                xanchor="center",
            ),
            yaxis_title="velocity (CFS)",
            xaxis_title="Month",
            yaxis=boxplot_config.yaxis,
            xaxis=boxplot_config.xaxis,
        )

        violin_plot = px.violin(
            ops_data,
            x="month_name",
            y="velocity",
            color="gate",
            box=True,
            category_orders={"month_name": list(month_names.values())},
            title="Velocity Distribution (Violin Plot)",
        )
        violin_plot.update_layout(
            yaxis=violin_config.yaxis,
            xaxis=violin_config.xaxis,
            legend=dict(
                orientation="h",
                y=1.1,
                x=0.5,
                xanchor="center",
            ),
        )
        gates = ops_data['gate'].unique()
        ops_data_by_gate = {gate: ops_data[ops_data['gate'] == gate] for gate in gates}
        ops_data_filtered_by_velocity = {gate:ops_data_by_gate[gate][ops_data_by_gate[gate]['is_over_8fs']] for gate in ops_data_by_gate}
        v_hist_charts = {gate: create_velocity_hist_chart(df, gate) for gate, df in ops_data_by_gate.items()}
        streak_hist_charts = {gate: create_streak_hist_chart(df, gate) for gate, df in ops_data_filtered_by_velocity.items()}
        hydro_locations = ["dgl", "old", "mho"]
        hydro_data_by_gate = {location: hydro_data[hydro_data.node== location] for location in hydro_locations}
        elev_hist_charts = {location: create_elev_hist_chart(df, location) for location, df in hydro_data_by_gate.items()}
        
        st.plotly_chart(boxplot, use_container_width=True, key=boxplot_config.key)
        st.plotly_chart(violin_plot, use_container_width=True, key=violin_config.key)
        col1, col2, col3 = st.columns(3)
        with col1:
            print(ops_data.columns)
            st.altair_chart(v_hist_charts[gates[0]], use_container_width=True)
            st.altair_chart(streak_hist_charts[gates[0]], use_container_width=True)
            st.altair_chart(elev_hist_charts[hydro_locations[0]], use_container_width=True)
        with col2:
            st.altair_chart(v_hist_charts[gates[1]], use_container_width=True)
            st.altair_chart(streak_hist_charts[gates[1]], use_container_width=True)
            st.altair_chart(elev_hist_charts[hydro_locations[1]], use_container_width=True)
        with col3:
            st.altair_chart(v_hist_charts[gates[2]], use_container_width=True)
            st.altair_chart(streak_hist_charts[gates[2]], use_container_width=True)
            st.altair_chart(elev_hist_charts[hydro_locations[2]], use_container_width=True)
    else:
        st.warning(f"No data available for May-November period in this scenario")


def generate_vel_gate_data(scenario_data, scenario):
    gate_data = scenario_data["gate_operations"].sort_values(by="datetime")
    gate_data_in_op_season = gate_data[gate_data["datetime"].dt.month.between(5, 11)]
    gate_data_in_op_season["gate_status"] = gate_data_in_op_season["value"] >= 10
    gate_data_in_op_season["consecutive_groups"] = gate_data_in_op_season.groupby(
        "node"
    )["value"].transform(lambda x: (x != x.shift()).cumsum())
    gate_data_in_op_season["group_min_datetime"] = gate_data_in_op_season.groupby(
        ["node", "consecutive_groups"]
    )["datetime"].transform(min)
    gate_data_in_op_season["group_max_datetime"] = gate_data_in_op_season.groupby(
        ["node", "consecutive_groups"]
    )["datetime"].transform(max)

    consecutive_streaks = (
        gate_data_in_op_season.groupby(
            [
                "node",
                "consecutive_groups",
                "value",
                "group_min_datetime",
                "group_max_datetime",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    consecutive_streaks["duration"] = consecutive_streaks["count"] * 15 / 60
    consecutive_streaks = consecutive_streaks.drop(
        ["value", "consecutive_groups"], axis=1
    )

    node_to_names_maps = {
        "old_gateop": "Old River",
        "glc_gateop": "Grantline",
        "mid_gateop": "Middle River",
        "glc": "Grantline",
        "old": "Old River",
        "mid": "Middle River",
    }
    gate_data_final = (
        pd.merge(
            gate_data_in_op_season,
            consecutive_streaks,
            left_on=["group_min_datetime", "group_max_datetime", "node"],
            right_on=["group_min_datetime", "group_max_datetime", "node"],
        )
        .drop(["consecutive_groups", "value"], axis=1)
        .rename(
            columns={
                "group_min_datetime": "gate_min_datetime",
                "group_max_datetime": "gate_max_datetime",
                "count": "gate_count",
                "duration": "gate_streak_duration",
            }
        )
        .assign(gate=lambda x: x["node"].map(node_to_names_maps))
    )

    velocity_data = scenario_data["vel"].sort_values(by="datetime")
    velocity_data["year"] = velocity_data["datetime"].dt.year
    velocity_data["is_over_8fs"] = velocity_data["vel"] >= 8
    velocity_data["consecutive_groups"] = velocity_data.groupby("location")[
        "is_over_8fs"
    ].transform(lambda x: (x != x.shift()).cumsum())

    velocity_data["group_min_datetime"] = velocity_data.groupby(
        ["location", "consecutive_groups"]
    )["datetime"].transform(min)

    velocity_data["group_max_datetime"] = velocity_data.groupby(
        ["location", "consecutive_groups"]
    )["datetime"].transform(max)

    velocity_data["date"] = velocity_data["datetime"].dt.date.astype(str)

    velocity_consecutive_streaks = (
        velocity_data.groupby(
            [
                "location",
                "consecutive_groups",
                "is_over_8fs",
                "group_min_datetime",
                "group_max_datetime",
            ]
        )
        .size()
        .reset_index(name="count")
    )
    velocity_consecutive_streaks["duration"] = (
        velocity_consecutive_streaks["count"] * 15 / 60
    )

    velocity_consecutive_streaks = velocity_consecutive_streaks.drop(
        columns=["consecutive_groups", "is_over_8fs", "group_max_datetime"]
    )

    velocity_data_final = (
        pd.merge(
            velocity_data,
            velocity_consecutive_streaks,
            left_on=["location", "group_min_datetime"],
            right_on=["location", "group_min_datetime"],
        )
        .rename(columns={"vel": "velocity"})
        .assign(gate=lambda x: x["location"].map(node_to_names_maps))
    )

    # TODO: wny are these merged?
    # merge gate and velocity datasets
    ops_data = pd.merge(
        gate_data_final,
        velocity_data_final,
        left_on=["datetime", "gate"],
        right_on=["datetime", "gate"],
    )

    ops_data["time_unit"] = 0.25
    ops_data["gate_status"] = np.where(ops_data["gate_status"], "Closed", "Open")
    ops_data["week"] = ops_data["datetime"].dt.isocalendar().week
    ops_data["model"] = scenario

    # daily metrics for the velocity
    daily_velocity = (
        ops_data.groupby(["gate", "date", "is_over_8fs"])["time_unit"]
        .sum()
        .reset_index()
    )
    avg_daily_velocity = pd.DataFrame(
        daily_velocity.groupby(["gate", "is_over_8fs"])["time_unit"].sum()
        / daily_velocity["date"].nunique()
    ).reset_index()
    daily_gate = (
        ops_data.groupby(["gate", "date", "gate_status"])["time_unit"]
        .sum()
        .reset_index()
    )
    avg_daily_gate = pd.DataFrame(
        daily_gate.groupby(["gate", "gate_status"])["time_unit"].sum()
        / daily_gate["date"].nunique()
    ).reset_index()
    daily_velocity_stats = (
        ops_data.groupby(["gate", "date", "is_over_8fs"])
        .agg(
            unique_consecutive_groups=("consecutive_groups", "nunique"),
            total_time=("time_unit", "sum"),
        )
        .reset_index()
    )
    daily_velocity_stats["daily_average_time_per_consecutive_group"] = (
        daily_velocity_stats["total_time"]
        / daily_velocity_stats["unique_consecutive_groups"]
    )
    total_daily_velocity = (
        daily_velocity_stats.groupby(["gate", "is_over_8fs"])[
            "daily_average_time_per_consecutive_group"
        ]
        .mean()
        .reset_index()
    )
    daily_gate_stats = (
        ops_data.groupby(["gate", "date", "gate_status"])
        .agg(
            unique_gate_count=("gate_count", "nunique"), total_time=("time_unit", "sum")
        )
        .reset_index()
    )
    daily_gate_stats["daily_average_time_per_consecutive_gate"] = (
        daily_gate_stats["total_time"] / daily_gate_stats["unique_gate_count"]
    )
    total_daily_gate = (
        daily_gate_stats.groupby(["gate", "gate_status"])[
            "daily_average_time_per_consecutive_gate"
        ]
        .mean()
        .reset_index()
    )

    hydro_data = scenario_data["water_levels"].sort_values(by="datetime")
    hydro_data = hydro_data[hydro_data["datetime"].dt.month.between(5, 11)]
    hydro_data["time_unit"] = 0.25
    hydro_data = hydro_data.rename(columns={"value": "water_level"})
    hydro_data["week"] = hydro_data["datetime"].dt.isocalendar().week

    return {
        "ops_data": ops_data,
        "hydro_data": hydro_data,
        "avg_daily_velocity": avg_daily_velocity,
        "avg_daily_gate": avg_daily_gate,
        "total_daily_velocity": total_daily_velocity,
        "total_daily_gate": total_daily_gate,
    }


st.title("Scenario Comparison")

if "previous_year_1" not in st.session_state:
    st.session_state.previous_year_1 = None
if "previous_scenario_1" not in st.session_state:
    st.session_state.previous_scenario_1 = None
if "scenario_data_1" not in st.session_state:
    st.session_state.scenario_data_1 = None
    st.session_state.scenario_year_data_1 = None
if "available_years_1" not in st.session_state:
    st.session_state.available_years_1 = []

if "previous_year_2" not in st.session_state:
    st.session_state.previous_year_2 = None
if "previous_scenario_2" not in st.session_state:
    st.session_state.previous_scenario_2 = None
if "scenario_data_2" not in st.session_state:
    st.session_state.scenario_data_2 = None
    st.session_state.scenario_year_data_2 = None
if "available_years_2" not in st.session_state:
    st.session_state.available_years_2 = []

scenarios = get_all_scenarios()
scenario_list = scenarios["Scenario"].tolist()

col1, col2 = st.columns(2)

with col1:
    selected_model_left, selected_year_left = get_scenario_and_year_selection(
        column="1",
        scenario_key="scenario_select_1",
        year_key="year_select_1",
        previous_scenario_key="previous_scenario_1",
        available_years_key="available_years_1",
        previous_year_key="previous_year_1",
    )

with col2:
    selected_model_right, selected_year_right = get_scenario_and_year_selection(
        column="2",
        scenario_key="scenario_select_2",
        year_key="year_select_2",
        previous_scenario_key="previous_scenario_2",
        available_years_key="available_years_2",
        previous_year_key="previous_year_2",
    )


submit_button = st.button("Submit")

if submit_button:
    st.checkbox("Show as difference from baseline", value=False)
    scenario_year_data_left, scenario_data_left = load_scenario_data(
        selected_model_left, selected_year_left
    )
    scenario_year_data_right, scenario_data_right = load_scenario_data(
        selected_model_right, selected_year_right
    )

    node_list = ["glc_flow_fish", "old_flow_fish", "mid_flow_fish"]

    month_names = {
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
    }

    left_data = generate_vel_gate_data(scenario_data_left, selected_model_left)
    right_data = generate_vel_gate_data(scenario_data_right, selected_model_right)
    left_hydro_data = full_post_process_hydro_data(scenario_data_left, selected_model_left)
    right_hydro_data = full_post_process_hydro_data(scenario_data_right, selected_model_right)

    left_ops_data = left_data["ops_data"].sort_values(by=["gate", "datetime"])
    left_min_dates = left_ops_data.groupby("gate")["datetime"].min().reset_index()
    left_max_dates = left_ops_data.groupby("gate")["datetime"].max().reset_index()

    right_ops_data = right_data["ops_data"]
    right_min_dates = right_ops_data.groupby("gate")["datetime"].min().reset_index()
    right_max_dates = right_ops_data.groupby("gate")["datetime"].max().reset_index()

    right_ops_data["month_name"] = right_ops_data["datetime"].dt.month.map(month_names)
    left_ops_data["month_name"] = left_ops_data["datetime"].dt.month.map(month_names)

    violin_ranges = [
        min(
            left_ops_data[["velocity"]].min().iloc[0] - 2,
            right_ops_data[["velocity"]].min().iloc[0] - 2,
        ),
        max(
            left_ops_data[["velocity"]].max().iloc[0] + 2,
            right_ops_data[["velocity"]].max().iloc[0] + 2,
        ),
    ]

    boxplot_ranges = [
        min(
            left_ops_data[["velocity"]].min().iloc[0] - 2,
            right_ops_data[["velocity"]].min().iloc[0] - 2,
        ),
        max(
            left_ops_data[["velocity"]].max().iloc[0] + 2,
            right_ops_data[["velocity"]].max().iloc[0] + 2,
        ),
    ]

    global_box_yaxis = dict(range=boxplot_ranges, showline=True, ticklabelstandoff=5)
    global_violin_yaxis = dict(range=violin_ranges, showline=True, ticklabelstandoff=5)
    global_xaxis = dict(showline=True)

    with col1:
        boxplot_config = ScenarioPlotConfig(
            key="left_boxplot",
            yaxis=global_box_yaxis,
            xaxis=global_xaxis,
        )
        violin_config = ScenarioPlotConfig(
            key="left_violin",
            yaxis=global_violin_yaxis,
            xaxis=global_xaxis,
        )
        render_scenario(
            left_data,
            left_hydro_data,
            selected_model_left,
            selected_year_left,
            month_names=month_names,
            boxplot_config=boxplot_config,
            violin_config=violin_config,
        )

    with col2:
        boxplot_config = ScenarioPlotConfig(
            key="right_boxplot",
            yaxis=global_box_yaxis,
            xaxis=global_xaxis,
        )
        violin_config = ScenarioPlotConfig(
            key="right_violin",
            yaxis=global_violin_yaxis,
            xaxis=global_xaxis,
        )
        render_scenario(
            right_data,
            right_hydro_data,
            selected_model_right,
            selected_year_right,
            month_names=month_names,
            boxplot_config=boxplot_config,
            violin_config=violin_config,
        )
else:
    st.write("Please select the scenarios and years, then click 'Submit' view data")
