import streamlit as st
import pandas as pd
import altair as alt
import pickle
from figures_functions import *
import plotly.figure_factory as ff
import plotly.express as px


# import plotly.express as px
import datetime
import numpy as np
from db import (
    get_scenario_year_data,
    get_all_scenarios,
    generate_scenario_year_data,
    get_filter_nodes_for_gate,
    get_available_years,
)


@st.cache_data
def load_scenario_data(scenario, year):
    """Cache the scenario data loading to prevent unnecessary database queries"""
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
    """Function to handle the scenario selection and dynamic year loading."""

    # Scenario selection
    selected_model = st.selectbox(f"Select Scenario:", scenario_list, key=scenario_key)

    # Load years dynamically based on selected scenario
    if selected_model != st.session_state[previous_scenario_key]:
        st.session_state[available_years_key] = get_available_years(selected_model)
        st.session_state[previous_scenario_key] = selected_model
        st.session_state[previous_year_key] = None  # Reset year when scenario changes

    # Year selection
    if st.session_state[available_years_key]:
        selected_year = st.selectbox(
            f"Select Year:", st.session_state[available_years_key], key=year_key
        )
        # st.write(f"Selected Scenario: {selected_model}, Year: {selected_year}")
    else:
        selected_year = None

    return selected_model, selected_year


def generate_vel_gate_data(
    scenario_data, selected_model, selected_year, gate_name, stream_gate
):
    """
    Processes gate operations, velocity, and water level data for a given gate.

    Args:
        scenario_data (dict): Dictionary containing scenario data with keys 'gate_operations', 'vel', and 'water_levels'.
        selected_model (str): The model selected for processing.
        selected_year (int): The year to filter the data.
        gate_name (str, optional): The name of the gate to process. Defaults to "glc".

    Returns:
        dict: A dictionary containing processed DataFrames:
            - full_merged_df: Merged DataFrame with gate operations and velocity data.
            - hydro_df: Processed hydro data DataFrame.
            - avg_daily_velocity: DataFrame with average daily velocity.
            - avg_daily_gate: DataFrame with average daily gate operations.
            - total_daily_velocity: DataFrame with consecutive velocity data.
            - total_daily_gate: DataFrame with consecutive gate operation data.
    """
    # Filter gate operations and velocity data based on node/location filters
    gate_data = scenario_data["gate_operations"]
    gate_data = gate_data[
        gate_data["node"] == get_filter_nodes_for_gate(gate_name, "gate_operations")
    ]

    vel_data = scenario_data["vel"]
    vel_data = vel_data[
        vel_data["location"] == get_filter_nodes_for_gate(gate_name, "velocity")
    ]

    # Merge and process full data
    full_merged_df = post_process_full_data(
        gate_data, vel_data, selected_model, gate_name, year=selected_year
    )
    full_merged_df = full_merged_df.rename(columns={"vel": "velocity"})

    # Process hydro data
    hydro_df = post_process_hydro_data(
        scenario_data["water_levels"], selected_model, stream_gate, selected_year
    )

    # Calculate daily averages and consecutive lengths
    avg_daily_velocity = calc_avg_daily_vel(full_merged_df)
    avg_daily_gate = calc_avg_daily_gate(full_merged_df)
    total_daily_velocity = calc_avg_len_consec_vel(full_merged_df)
    total_daily_gate = calc_avg_len_consec_gate(full_merged_df)

    return {
        "full_merged_df": full_merged_df,
        "hydro_df": hydro_df,
        "avg_daily_velocity": avg_daily_velocity,
        "avg_daily_gate": avg_daily_gate,
        "total_daily_velocity": total_daily_velocity,
        "total_daily_gate": total_daily_gate,
    }


# Initialize Streamlit app
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

# Step 1: Load available scenarios first
scenarios = get_all_scenarios()
scenario_list = scenarios["Scenario"].tolist()

# Step 2: Create two columns
col1, col2 = st.columns(2)

# Step 3: Call function for both columns
with col1:
    selected_model_1, selected_year_1 = get_scenario_and_year_selection(
        column="1",
        scenario_key="scenario_select_1",
        year_key="year_select_1",
        previous_scenario_key="previous_scenario_1",
        available_years_key="available_years_1",
        previous_year_key="previous_year_1",
    )

with col2:
    selected_model_2, selected_year_2 = get_scenario_and_year_selection(
        column="2",
        scenario_key="scenario_select_2",
        year_key="year_select_2",
        previous_scenario_key="previous_scenario_2",
        available_years_key="available_years_2",
        previous_year_key="previous_year_2",
    )

submit_button = st.button("Submit")

if submit_button:
    scenario_year_data_1, scenario_data_1 = load_scenario_data(
        selected_model_1, selected_year_1
    )
    scenario_year_data_2, scenario_data_2 = load_scenario_data(
        selected_model_2, selected_year_2
    )

    node_list = ["glc_flow_fish", "old_flow_fish", "mid_flow_fish"]

    node_data_1 = scenario_year_data_1[scenario_year_data_1["node"].isin(node_list)]
    node_data_1["datetime"] = pd.to_datetime(node_data_1["datetime"])
    node_data_1_filtered = node_data_1[
        (node_data_1["datetime"].dt.month >= 5)
        & (node_data_1["datetime"].dt.month <= 11)
    ]

    node_data_2 = scenario_year_data_2[scenario_year_data_2["node"].isin(node_list)]
    node_data_2["datetime"] = pd.to_datetime(node_data_2["datetime"])
    node_data_2_filtered = node_data_2[
        (node_data_2["datetime"].dt.month >= 5)
        & (node_data_2["datetime"].dt.month <= 11)
    ]

    month_names = {
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
    }

    # node_data_1_filtered["month_name"] = node_data_1_filtered["datetime"].dt.month.map(
    #     month_names
    # )
    # node_data_2_filtered["month_name"] = node_data_2_filtered["datetime"].dt.month.map(
    #     month_names
    # )

    if not node_data_1_filtered.empty and not node_data_2_filtered.empty:
        min_y = min(
            node_data_1_filtered["value"].min(), node_data_2_filtered["value"].min()
        )
        max_y = max(
            node_data_1_filtered["value"].max(), node_data_2_filtered["value"].max()
        )

        y_range_padding = (max_y - min_y) * 0.05
        y_min = min_y - y_range_padding
        y_max = max_y + y_range_padding

        y_range = [y_min, y_max]
    else:
        y_range = None

    with col1:
        st.success(f"Scenario 1 Loaded: {selected_model_1} ({selected_year_1})")
        st.write(scenario_data_1["flow"], use_container_width=True)

        # if not node_data_1_filtered.empty:
        #     fig1 = px.box(
        #         node_data_1_filtered,
        #         x="month_name",
        #         y="value",
        #         color="node",
        #         category_orders={"month_name": list(month_names.values())},
        #         title=f"Flow Distribution by Month (May-Nov) - {selected_model_1} ({selected_year_1})",
        #         labels={
        #             "month_name": "Month",
        #             "value": "Flow (CFS)",
        #             "node": "Location",
        #         },
        #         height=500,
        #         points="outliers",
        #     )
        #
        #     fig1.update_traces(
        #         hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>Flow: %{y:.3f} CFS<extra></extra>",
        #         customdata=[[node] for node in node_data_1_filtered["node"]],
        #     )
        #
        #     fig1.update_layout(
        #         boxmode="group",
        #         legend=dict(
        #             orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        #         ),
        #         yaxis_title="Flow (CFS)",
        #         xaxis_title="Month",
        #         yaxis=dict(range=y_range),
        #     )
        #
        #     fig1_1 = px.violin(
        #         node_data_1_filtered,
        #         x="month_name",
        #         y="value",
        #         color="node",
        #         box=True,  # Show box plot inside violin
        #         category_orders={"month_name": list(month_names.values())},
        #         title="Flow Distribution (Violin Plot)",
        #     )
        #     fig1_1.update_layout(yaxis=dict(range=y_range))
        #
        #     st.plotly_chart(fig1, use_container_width=True)
        #     st.plotly_chart(fig1_1, use_container_width=True)
        # else:
        #     st.warning("No data available for May-November period in Scenario 1")

        glc_vel_gate_data = generate_vel_gate_data(
            scenario_data_1, selected_model_1, selected_year_1, "glc", "dgl"
        )
        old_vel_gate_data = generate_vel_gate_data(
            scenario_data_1, selected_model_1, selected_year_1, "old", "old"
        )
        mid_vel_gate_data = generate_vel_gate_data(
            scenario_data_1, selected_model_1, selected_year_1, "mid", "mho"
        )
        glc_min_date = min(glc_vel_gate_data["full_merged_df"]["date"])
        glc_max_date = max(glc_vel_gate_data["full_merged_df"]["date"])

        velocity_summary_stats_title = (
            f"Summary stats of fish passage from {glc_min_date} to {glc_max_date}."
        )
        gate_summary_stats_title = (
            f"Summary stats of upstream of gate from {glc_min_date} to {glc_max_date}."
        )
        min_max_summary_title = (
            f"Min max stats of fish passage from {glc_min_date} to {glc_max_date}."
        )

        velocity_summary_data = {
            "Location": [
                location_gate[glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            f"Average Daily Time (Hours) {glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][0]}": [
                round(glc_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(mid_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(old_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
            ],
            f"Average Streak Duration (Hours) {glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][0]}": [
                round(
                    glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
            ],
        }
        if len(glc_vel_gate_data["avg_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Daily Time (Hours) {glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(glc_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(glc_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(mid_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(mid_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(old_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(old_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
            ]
        if len(glc_vel_gate_data["total_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Streak Duration (Hours) {glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(
                    glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
            ]

            gate_summary_data = {
                "Location": [
                    location_gate[glc_vel_gate_data["full_merged_df"]["gate"][0]],
                    location_gate[mid_vel_gate_data["full_merged_df"]["gate"][0]],
                    location_gate[old_vel_gate_data["full_merged_df"]["gate"][0]],
                ],
                f"Average Daily {glc_vel_gate_data["avg_daily_gate"]['gate_status'][0]} Time (Hours) for gate": [
                    round(glc_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                    round(mid_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                    round(old_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                ],
                f"Average {glc_vel_gate_data["total_daily_gate"]['gate_status'][0]} Duration (Hours) Per Streak": [
                    round(
                        glc_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                    round(
                        mid_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                    round(
                        old_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                ],
            }
        if len(glc_vel_gate_data["avg_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average Daily {glc_vel_gate_data["avg_daily_gate"]['gate_status'][1]} Time (Hours) for gate"
            ] = [
                round(glc_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(glc_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(mid_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(mid_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(old_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(old_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
            ]

        if len(glc_vel_gate_data["total_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average {glc_vel_gate_data["total_daily_gate"]['gate_status'][1]} Duration (Hours) Per Streak"
            ] = [
                round(
                    glc_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    mid_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    old_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
            ]

        min_max_summary = {
            "Location": [
                location_gate[glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            "Minimum velocity through fish passage (ft/s)": [
                round(min(glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
            "Maximum velocity through fish passage (ft/s)": [
                round(max(glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
        }

        # Create a DataFrame
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

        # st.dataframe(glc_vel_gate_data["full_merged_df"])
        all_vel_data_1 = pd.concat(
            [
                glc_vel_gate_data["full_merged_df"],
                old_vel_gate_data["full_merged_df"],
                mid_vel_gate_data["full_merged_df"],
            ]
        )
        all_vel_data_1["month_name"] = all_vel_data_1["datetime"].dt.month.map(
            month_names
        )

        if not all_vel_data_1.empty:
            fig1 = px.box(
                all_vel_data_1,
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

            fig1.update_traces(
                hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>velocity: %{y:.3f} CFS<extra></extra>",
                customdata=[[node] for node in node_data_1_filtered["node"]],
            )

            fig1.update_layout(
                boxmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                yaxis_title="velocity (CFS)",
                xaxis_title="Month",
                yaxis=dict(range=[-10, 30]),
            )

            fig1_1 = px.violin(
                all_vel_data_1,
                x="month_name",
                y="velocity",
                color="location",
                box=True,  # Show box plot inside violin
                category_orders={"month_name": list(month_names.values())},
                title="Velocity Distribution (Violin Plot)",
            )
            fig1_1.update_layout(yaxis=dict(range=[-10, 30]))

            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig1_1, use_container_width=True)
        else:
            st.warning("No data available for May-November period in Scenario 1")

    with col2:
        st.success(f"Scenario 2 Loaded: {selected_model_2} ({selected_year_2})")
        st.dataframe(scenario_data_2["flow"], use_container_width=True)

        # if not node_data_2_filtered.empty:
        #     fig2 = px.box(
        #         node_data_2_filtered,
        #         x="month_name",
        #         y="value",
        #         color="node",
        #         category_orders={"month_name": list(month_names.values())},
        #         title=f"Flow Distribution by Month (May-Nov) - {selected_model_2} ({selected_year_2})",
        #         labels={
        #             "month_name": "Month",
        #             "value": "Flow (CFS)",
        #             "node": "Location",
        #         },
        #         height=500,
        #         points="outliers",
        #     )
        #
        #     fig2.update_traces(
        #         hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>Flow: %{y:.1f} CFS<extra></extra>",
        #         customdata=[[node] for node in node_data_2_filtered["node"]],
        #     )
        #
        #     fig2.update_layout(
        #         boxmode="group",
        #         legend=dict(
        #             orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        #         ),
        #         yaxis_title="Flow (CFS)",
        #         xaxis_title="Month",
        #         yaxis=dict(range=None),
        #     )
        #
        #     fig2_2 = px.violin(
        #         node_data_2_filtered,
        #         x="month_name",
        #         y="value",
        #         color="node",
        #         box=True,  # Show box plot inside violin
        #         category_orders={"month_name": list(month_names.values())},
        #         title="Flow Distribution (Violin Plot)",
        #     )
        #     fig2_2.update_layout(yaxis=dict(range=None))
        #
        #     st.plotly_chart(fig2, use_container_width=True)
        #     st.plotly_chart(
        #         fig2_2,
        #         use_container_width=True,
        #     )

        # else:
        #     st.warning("No data available for May-November period in Scenario 2")
        glc_vel_gate_data = generate_vel_gate_data(
            scenario_data_2, selected_model_2, selected_year_2, "glc", "dgl"
        )
        old_vel_gate_data = generate_vel_gate_data(
            scenario_data_2, selected_model_2, selected_year_2, "old", "old"
        )
        mid_vel_gate_data = generate_vel_gate_data(
            scenario_data_2, selected_model_2, selected_year_2, "mid", "mho"
        )
        glc_min_date = min(glc_vel_gate_data["full_merged_df"]["date"])
        glc_max_date = max(glc_vel_gate_data["full_merged_df"]["date"])

        velocity_summary_stats_title = (
            f"Summary stats of fish passage from {glc_min_date} to {glc_max_date}."
        )
        gate_summary_stats_title = (
            f"Summary stats of upstream of gate from {glc_min_date} to {glc_max_date}."
        )
        min_max_summary_title = (
            f"Min max stats of fish passage from {glc_min_date} to {glc_max_date}."
        )

        velocity_summary_data = {
            "Location": [
                location_gate[glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            f"Average Daily Time (Hours) {glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][0]}": [
                round(glc_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(mid_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
                round(old_vel_gate_data["avg_daily_velocity"]["time_unit"][0], 2),
            ],
            f"Average Streak Duration (Hours) {glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][0]}": [
                round(
                    glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
                round(
                    old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][0],
                    2,
                ),
            ],
        }
        if len(glc_vel_gate_data["avg_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Daily Time (Hours) {glc_vel_gate_data["avg_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(glc_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(glc_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(mid_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(mid_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
                round(old_vel_gate_data["avg_daily_velocity"]["time_unit"][1], 2)
                if len(old_vel_gate_data["avg_daily_velocity"]["time_unit"]) > 1
                else 0,
            ]
        if len(glc_vel_gate_data["total_daily_velocity"]["Velocity_Category"]) > 1:
            velocity_summary_data[
                f"Average Streak Duration (Hours) {glc_vel_gate_data["total_daily_velocity"]['Velocity_Category'][1]}"
            ] = [
                round(
                    glc_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    mid_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
                round(
                    old_vel_gate_data["total_daily_velocity"][
                        "daily_average_time_per_consecutive_group"
                    ][1],
                    2,
                ),
            ]

            gate_summary_data = {
                "Location": [
                    location_gate[glc_vel_gate_data["full_merged_df"]["gate"][0]],
                    location_gate[mid_vel_gate_data["full_merged_df"]["gate"][0]],
                    location_gate[old_vel_gate_data["full_merged_df"]["gate"][0]],
                ],
                f"Average Daily {glc_vel_gate_data["avg_daily_gate"]['gate_status'][0]} Time (Hours) for gate": [
                    round(glc_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                    round(mid_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                    round(old_vel_gate_data["avg_daily_gate"]["time_unit"][0], 2),
                ],
                f"Average {glc_vel_gate_data["total_daily_gate"]['gate_status'][0]} Duration (Hours) Per Streak": [
                    round(
                        glc_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                    round(
                        mid_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                    round(
                        old_vel_gate_data["total_daily_gate"][
                            "daily_average_time_per_consecutive_gate"
                        ][0],
                        2,
                    ),
                ],
            }
        if len(glc_vel_gate_data["avg_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average Daily {glc_vel_gate_data["avg_daily_gate"]['gate_status'][1]} Time (Hours) for gate"
            ] = [
                round(glc_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(glc_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(mid_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(mid_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
                round(old_vel_gate_data["avg_daily_gate"]["time_unit"][1], 2)
                if len(old_vel_gate_data["avg_daily_gate"]["time_unit"]) > 1
                else 0,
            ]

        if len(glc_vel_gate_data["total_daily_gate"]["gate_status"]) > 1:
            gate_summary_data[
                f"Average {glc_vel_gate_data["total_daily_gate"]['gate_status'][1]} Duration (Hours) Per Streak"
            ] = [
                round(
                    glc_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    mid_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
                round(
                    old_vel_gate_data["total_daily_gate"][
                        "daily_average_time_per_consecutive_gate"
                    ][1],
                    2,
                ),
            ]

        min_max_summary = {
            "Location": [
                location_gate[glc_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[mid_vel_gate_data["full_merged_df"]["gate"][0]],
                location_gate[old_vel_gate_data["full_merged_df"]["gate"][0]],
            ],
            "Minimum velocity through fish passage (ft/s)": [
                round(min(glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(min(old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
            "Maximum velocity through fish passage (ft/s)": [
                round(max(glc_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(mid_vel_gate_data["full_merged_df"]["velocity"]), 2),
                round(max(old_vel_gate_data["full_merged_df"]["velocity"]), 2),
            ],
        }

        # Create a DataFrame
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

        all_vel_data_2 = pd.concat(
            [
                glc_vel_gate_data["full_merged_df"],
                old_vel_gate_data["full_merged_df"],
                mid_vel_gate_data["full_merged_df"],
            ]
        )

        all_vel_data_2["month_name"] = all_vel_data_2["datetime"].dt.month.map(
            month_names
        )

        if not all_vel_data_2.empty:
            fig2 = px.box(
                all_vel_data_2,
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

            fig2.update_traces(
                hovertemplate="<b>%{x}</b><br>Location: %{customdata}<br>velocity: %{y:.3f} CFS<extra></extra>",
                customdata=[[node] for node in all_vel_data_2["location"]],
            )

            fig2.update_layout(
                boxmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                yaxis_title="velocity (CFS)",
                xaxis_title="Month",
                yaxis=dict(range=[-10, 30]),
            )

            fig2_1 = px.violin(
                all_vel_data_2,
                x="month_name",
                y="velocity",
                color="location",
                box=True,  # Show box plot inside violin
                category_orders={"month_name": list(month_names.values())},
                title="Velocity Distribution (Violin Plot)",
            )
            fig2_1.update_layout(yaxis=dict(range=[-10, 30]))

            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig2_1, use_container_width=True)
        else:
            st.warning("No data available for May-November period in Scenario 1")

else:
    st.write(
        "Please select the scenarios and years, then click 'Submit' to preview the data."
    )
