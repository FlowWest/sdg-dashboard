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


@st.cache_data
def load_scenario_data(scenario, year):
    scenario_year_data = get_scenario_year_data(scenario, year)
    scenario_data = generate_scenario_year_data(scenario_year_data)
    return scenario_year_data, scenario_data


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

    violin_ranges = None
    boxplot_ranges = None

    left_data = generate_vel_gate_data(scenario_data_left, selected_model_left)
    right_data = generate_vel_gate_data(scenario_data_right, selected_model_right)

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

    with col1:
        left_scenario_date_range = left_ops_data["date"].agg(["min", "max"]).tolist()
        st.success(f"Scenario 1 Loaded: {selected_model_left} ({selected_year_left})")

        # TODO: finish formatting this table so that its the best it can possibly be
        st.dataframe(
            left_ops_data,
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

        velocity_summary_stats_title = f"Summary stats of fish passage from {left_scenario_date_range[0]} to {left_scenario_date_range[1]}."
        gate_summary_stats_title = f"Summary stats of upstream of gate from {left_scenario_date_range[0]} to {left_scenario_date_range[1]}."
        min_max_summary_title = f"Min max stats of fish passage from {left_scenario_date_range[0]} to {left_scenario_date_range[1]}."

        velocity_summary_data = {
            "Location": [
                location_gate[left_glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[left_mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[left_old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            f"Average Daily Time (Hours) {left_glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][0]}": [
                round(left_glc_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(left_mid_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(left_old_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
            ],
            f"Average Streak Duration (Hours) {left_glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][0]}": [
                round(
                    left_glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    left_mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    left_old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
            ],
        }
        if len(left_glc_vel_gate_data["avg_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Daily Time (Hours) {left_glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(left_glc_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(left_glc_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(left_mid_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(left_mid_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(left_old_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(left_old_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
            ]
        if len(left_glc_vel_gate_data["total_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Streak Duration (Hours) {left_glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(
                    left_glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    left_mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    left_old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
            ]

        gate_summary_data = {
            "Location": [
                location_gate[left_glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[left_mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[left_old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            f"Average Daily {left_glc_vel_gate_data["avg_daily_gate"]['gate_status'][0]} Time (Hours) for gate": [
                round(left_glc_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                round(left_mid_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                round(left_old_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
            ],
            f"Average {left_glc_vel_gate_data["total_daily_gate"]['gate_status'][0]} Duration (Hours) Per Streak": [
                round(
                    left_glc_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][0],
                    2,
                ),
                round(
                    left_mid_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][0],
                    2,
                ),
                round(
                    left_old_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][0],
                    2,
                ),
            ],
        }
        if len(left_glc_vel_gate_data["avg_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average Daily {left_glc_vel_gate_data["avg_daily_gate"]['gate_status'][1]} Time (Hours) for gate"
            ] = [
                round(left_glc_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(left_glc_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(left_mid_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(left_mid_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(left_old_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(left_old_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
            ]

        if len(left_glc_vel_gate_data["total_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average {left_glc_vel_gate_data["total_daily_gate"]['gate_status'][1]} Duration (Hours) Per Streak"
            ] = [
                round(
                    left_glc_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    left_mid_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    left_old_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
            ]

        min_max_summary = {
            "Location": [
                location_gate[left_glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[left_mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[left_old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            "Minimum velocity through fish passage (ft/s)": [
                round(min(left_glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(left_mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(left_old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
            "Maximum velocity through fish passage (ft/s)": [
                round(max(left_glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(left_mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(left_old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
        }

        velocity_summary_df = pd.DataFrame(velocity_summary_data)
        gate_summary_df = pd.DataFrame(gate_summary_data)
        min_max_vel_summary_df = pd.DataFrame(min_max_summary)

        st.dataframe(
            velocity_summary_df.style.highlight_max(
                subset=velocity_summary_df.columns[1:], color="#ffffc5"
            ).format(precision=2)
        )
        st.dataframe(
            min_max_vel_summary_df.style.highlight_max(
                subset=min_max_vel_summary_df.columns[1:], color="#ffffc5"
            ).format(precision=2)
        )
        st.dataframe(
            gate_summary_df.style.highlight_max(
                subset=gate_summary_df.columns[1:], color="#ffffc5"
            ).format(precision=2)
        )

        if not left_scenario_velocity.empty:
            left_boxplot = px.box(
                left_scenario_velocity,
                x="month_name",
                y="velocity",
                color="location",
                category_orders={"month_name": list(month_names.values())},
                title=f"Velocity by Month (May-Nov) - {selected_model_1} ({selected_year_1})",
                labels={
                    "month_name": "Month",
                    "value": "velocity",
                    "node": "Location",
                },
                height=500,
                points="outliers",
            )

            left_boxplot.update_traces(
                hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>velocity: %{y:.3f} CFS<extra></extra>",
                customdata=[[node] for node in left_scenario_velocity["location"]],
            )

            left_boxplot.update_layout(
                boxmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                yaxis_title="velocity (CFS)",
                xaxis_title="Month",
                yaxis=dict(range=boxplot_ranges),
            )

            left_violin = px.violin(
                left_scenario_velocity,
                x="month_name",
                y="velocity",
                color="location",
                box=True,  # Show box plot inside violin
                category_orders={"month_name": list(month_names.values())},
                title="Velocity Distribution (Violin Plot)",
            )
            left_violin.update_layout(yaxis=dict(range=violin_ranges))

            st.plotly_chart(left_boxplot, use_container_width=True, key="left_boxplot")
            st.plotly_chart(left_violin, use_container_width=True, key="left_boxplot")
        else:
            st.warning("No data available for May-November period in Scenario 1")

    with col2:
        st.success(f"Scenario 2 Loaded: {selected_model_2} ({selected_year_2})")
        st.dataframe(scenario_data_2["flow"], use_container_width=True)

        velocity_summary_stats_title = f"Summary stats of fish passage from {right_glc_min_date} to {left_glc_max_date}."
        gate_summary_stats_title = f"Summary stats of upstream of gate from {right_glc_min_date} to {left_glc_max_date}."
        min_max_summary_title = f"Min max stats of fish passage from {right_glc_min_date} to {left_glc_max_date}."

        velocity_summary_data = {
            "Location": [
                location_gate[right_glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[right_mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[right_old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            f"Average Daily Time (Hours) {right_glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][0]}": [
                round(right_glc_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(right_mid_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(right_old_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
            ],
            f"Average Streak Duration (Hours) {right_glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][0]}": [
                round(
                    right_glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    right_mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    right_old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
            ],
        }
        if len(right_glc_vel_gate_data["avg_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Daily Time (Hours) {right_glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(right_glc_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(right_glc_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(right_mid_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(right_mid_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(right_old_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(right_old_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
            ]
        if (
            len(right_glc_vel_gate_data["total_daily_velocity"]["Velocity_Category"])
            > 1
        ):
            velocity_summary_data[
                f"Average Streak Duration (Hours) {right_glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(
                    right_glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    right_mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    right_old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
            ]

            gate_summary_data = {
                "Location": [
                    location_gate[right_glc_vel_gate_data["full_merged_df"]["gate"][0]],
                    location_gate[right_mid_vel_gate_data["full_merged_df"]["gate"][0]],
                    location_gate[right_old_vel_gate_data["full_merged_df"]["gate"][0]],
                ],
                f"Average Daily {right_glc_vel_gate_data["avg_daily_gate"]['gate_status'][0]} Time (Hours) for gate": [
                    round(right_glc_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                    round(right_mid_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                    round(right_old_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                ],
                f"Average {right_glc_vel_gate_data["total_daily_gate"]['gate_status'][0]} Duration (Hours) Per Streak": [
                    round(
                        right_glc_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                    round(
                        right_mid_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                    round(
                        right_old_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                ],
            }
        if len(right_glc_vel_gate_data["avg_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average Daily {right_glc_vel_gate_data["avg_daily_gate"]['gate_status'][1]} Time (Hours) for gate"
            ] = [
                round(right_glc_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(right_glc_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(right_mid_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(right_mid_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(right_old_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(right_old_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
            ]

        if len(right_glc_vel_gate_data["total_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average {right_glc_vel_gate_data["total_daily_gate"]['gate_status'][1]} Duration (Hours) Per Streak"
            ] = [
                round(
                    right_glc_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    right_mid_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    right_old_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
            ]

        min_max_summary = {
            "Location": [
                location_gate[right_glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[right_mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[right_old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            "Minimum velocity through fish passage (ft/s)": [
                round(min(right_glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(right_mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(right_old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
            "Maximum velocity through fish passage (ft/s)": [
                round(max(right_glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(right_mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(right_old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
        }

        velocity_summary_df = pd.DataFrame(velocity_summary_data)
        gate_summary_df = pd.DataFrame(gate_summary_data)
        min_max_vel_summary_df = pd.DataFrame(min_max_summary)

        st.dataframe(
            velocity_summary_df.style.highlight_max(
                subset=velocity_summary_df.columns[1:], color="#ffffc5"
            ).format(precision=2)
        )
        st.dataframe(
            min_max_vel_summary_df.style.highlight_max(
                subset=min_max_vel_summary_df.columns[1:], color="#ffffc5"
            ).format(precision=2)
        )
        st.dataframe(
            gate_summary_df.style.highlight_max(
                subset=gate_summary_df.columns[1:], color="#ffffc5"
            ).format(precision=2)
        )

        if not right_scenario_velocity.empty:
            right_boxplot = px.box(
                right_scenario_velocity,
                x="month_name",
                y="velocity",
                color="location",
                category_orders={"month_name": list(month_names.values())},
                title=f"Velocity by Month (May-Nov) - {selected_model_1} ({selected_year_1})",
                labels={
                    "month_name": "Month",
                    "value": "velocity",
                    "node": "Location",
                },
                height=500,
                points="outliers",
            )

            right_boxplot.update_traces(
                hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>velocity: %{y:.3f} CFS<extra></extra>",
                customdata=[[node] for node in right_scenario_velocity["location"]],
            )

            right_boxplot.update_layout(
                boxmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                yaxis_title="velocity (CFS)",
                xaxis_title="Month",
                yaxis=dict(range=boxplot_ranges),
            )

            right_violin = px.violin(
                right_scenario_velocity,
                x="month_name",
                y="velocity",
                color="location",
                box=True,  # Show box plot inside violin
                category_orders={"month_name": list(month_names.values())},
                title="Velocity Distribution (Violin Plot)",
            )
            right_violin.update_layout(yaxis=dict(range=violin_ranges))

            st.plotly_chart(right_boxplot, use_container_width=True, key="right_boc")
            st.plotly_chart(right_violin, use_container_width=True, key="right_violin")
        else:
            st.warning("No data available for May-November period in Scenario 1")

else:
    st.write("Please select the scenarios and years, then click 'Submit' view data")
