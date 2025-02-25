import streamlit as st
import pandas as pd
import altair as alt
import pickle
from figures_functions import *

# import plotly.express as px
import datetime
import numpy as np
from db import (
    get_scenario_year_data,
    get_all_scenarios,
    generate_scenario_year_data,
    get_filter_nodes_for_gate,
    get_available_years
)
# from streamlit_folium import st_folium, folium_static
#

@st.cache_data
def load_scenario_data(scenario, year):
    """Cache the scenario data loading to prevent unnecessary database queries"""
    scenario_year_data = get_scenario_year_data(scenario, year)
    scenario_data = generate_scenario_year_data(scenario_year_data)
    return scenario_year_data, scenario_data

st.title("Exploratory Data Visualizations for SDG Analysis")

# Initialize session state variables if not set
if "previous_year" not in st.session_state:
    st.session_state.previous_year = None
if "previous_scenario" not in st.session_state:
    st.session_state.previous_scenario = None
if "scenario_data" not in st.session_state:
    st.session_state.scenario_data = None
    st.session_state.scenario_year_data = None
if "available_years" not in st.session_state:
    st.session_state.available_years = []

# Step 1: Load available scenarios first
scenarios = get_all_scenarios()
scenario_list = scenarios["Scenario"].tolist()

# Step 2: Scenario selection (mandatory before year selection)
selected_model = st.sidebar.selectbox("Select Scenario:", scenario_list, key="scenario_select")

# Step 3: Load years dynamically based on selected scenario
if selected_model != st.session_state.previous_scenario:
    with st.spinner(f"Fetching available years for {selected_model}..."):
        # Assuming years are extracted dynamically from the scenario data
        st.session_state.available_years = get_available_years(selected_model)  
        st.session_state.previous_scenario = selected_model
        st.session_state.previous_year = None  # Reset year when scenario changes

# Step 4: Year selection (only after scenario is chosen)
if st.session_state.available_years:
    selected_year = st.sidebar.selectbox("Select Year:", st.session_state.available_years, key="year_select")
else:
    selected_year = None

# Step 5: Load scenario data only when both scenario & year are selected
if selected_model and selected_year:
    if (
        st.session_state.scenario_data is None
        or selected_year != st.session_state.previous_year
        or selected_model != st.session_state.previous_scenario
    ):
        with st.spinner(f"Loading {selected_model} data for {selected_year}..."):
            scenario_year_data, scenario_data = load_scenario_data(selected_model, selected_year)
            st.session_state.scenario_data = scenario_data
            st.session_state.scenario_year_data = scenario_year_data
            st.session_state.previous_year = selected_year

# Retrieve cached data
scenario_data = st.session_state.scenario_data
print(scenario_data)
scenario_year_data = st.session_state.scenario_year_data
print(scenario_year_data)
# --------------------------------------------------------------------------------------------------------------------------------
# Data wrangling
if scenario_data and not scenario_year_data.empty:
    # GLC wranging ------------------------------------
    glc_gate_data = scenario_data["gate_operations"]
    glc_gate_data = glc_gate_data[
        glc_gate_data["node"] == get_filter_nodes_for_gate("glc", "gate_operations")
    ]
    glc_vel_data = scenario_data["vel"]
    glc_vel_data = glc_vel_data[
        glc_vel_data["location"] == get_filter_nodes_for_gate("glc", "velocity")
    ]
    # print(glc_vel_data.columns)

    glc_full_merged_df = post_process_full_data(
        glc_gate_data, glc_vel_data, selected_model, "glc", year=selected_year
    )

    glc_full_merged_df = glc_full_merged_df.rename(columns={"vel": "velocity"})

    glc_hydro_df = post_process_hydro_data(
        scenario_data["water_levels"], selected_model, "dgl", selected_year
    )
    glc_avg_daily_velocity = calc_avg_daily_vel(glc_full_merged_df)
    glc_avg_daily_gate = calc_avg_daily_gate(glc_full_merged_df)
    glc_total_daily_velocity = calc_avg_len_consec_vel(glc_full_merged_df)
    glc_total_daily_gate = calc_avg_len_consec_gate(glc_full_merged_df)

    # OLD wranging ----------------------------------
    old_gate_data = scenario_data["gate_operations"]
    old_gate_data = old_gate_data[
        old_gate_data["node"] == get_filter_nodes_for_gate("old", "gate_operations")
    ]
    old_vel_data = scenario_data["vel"]
    old_vel_data = old_vel_data[
        old_vel_data["location"] == get_filter_nodes_for_gate("old", "velocity")
    ]

    old_full_merged_df = post_process_full_data(
        old_gate_data, old_vel_data, selected_model, "old", year=selected_year
    )

    old_full_merged_df = old_full_merged_df.rename(columns={"vel": "velocity"})

    old_hydro_df = post_process_hydro_data(
        scenario_data["water_levels"], selected_model, "old", selected_year
    )
    old_avg_daily_velocity = calc_avg_daily_vel(old_full_merged_df)
    old_avg_daily_gate = calc_avg_daily_gate(old_full_merged_df)
    old_total_daily_velocity = calc_avg_len_consec_vel(old_full_merged_df)
    old_total_daily_gate = calc_avg_len_consec_gate(old_full_merged_df)

    # MID wrangling -------------------------------------
    mid_gate_data = scenario_data["gate_operations"]
    mid_gate_data = mid_gate_data[
        mid_gate_data["node"] == get_filter_nodes_for_gate("mid", "gate_operations")
    ]
    mid_vel_data = scenario_data["vel"]
    mid_vel_data = mid_vel_data[
        mid_vel_data["location"] == get_filter_nodes_for_gate("mid", "velocity")
    ]

    mid_full_merged_df = post_process_full_data(
        mid_gate_data, mid_vel_data, selected_model, "mid", year=selected_year
    )

    mid_full_merged_df = mid_full_merged_df.rename(columns={"vel": "velocity"})

    mid_hydro_df = post_process_hydro_data(
        scenario_data["water_levels"], selected_model, "mho", selected_year
    )
    mid_avg_daily_velocity = calc_avg_daily_vel(mid_full_merged_df)
    mid_avg_daily_gate = calc_avg_daily_gate(mid_full_merged_df)
    mid_total_daily_velocity = calc_avg_len_consec_vel(mid_full_merged_df)
    mid_total_daily_gate = calc_avg_len_consec_gate(mid_full_merged_df)

    glc_min_date = min(glc_full_merged_df["date"])
    glc_max_date = max(glc_full_merged_df["date"])

    glc_min_velocity = min(glc_full_merged_df["velocity"])
    glc_max_velocity = max(glc_full_merged_df["velocity"])

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
            location_gate[glc_full_merged_df["gate"][0]],
            location_gate[mid_full_merged_df["gate"][0]],
            location_gate[old_full_merged_df["gate"][0]],
        ],
        f"Average Daily Time (Hours) {glc_avg_daily_velocity['Velocity_Category'][0]}": [
            round(glc_avg_daily_velocity["time_unit"][0], 2),
            round(mid_avg_daily_velocity["time_unit"][0], 2),
            round(old_avg_daily_velocity["time_unit"][0], 2),
        ],
        f"Average Streak Duration (Hours) {glc_total_daily_velocity['Velocity_Category'][0]}": [
        round(
            glc_total_daily_velocity["daily_average_time_per_consecutive_group"][0],
            2,
        ),
        round(
            mid_total_daily_velocity["daily_average_time_per_consecutive_group"][0],
            2,
        ),
        round(
            old_total_daily_velocity["daily_average_time_per_consecutive_group"][0],
            2,
        ),
        ]}
    if len(glc_avg_daily_velocity["Velocity_Category"]) > 1:
        velocity_summary_data[
                f"Average Daily Time (Hours) {glc_avg_daily_velocity['Velocity_Category'][1]}"
            ] = [
                round(glc_avg_daily_velocity["time_unit"][1], 2) if len(glc_avg_daily_velocity["time_unit"]) > 1 else 0,
                round(mid_avg_daily_velocity["time_unit"][1], 2) if len(mid_avg_daily_velocity["time_unit"]) > 1 else 0,
                round(old_avg_daily_velocity["time_unit"][1], 2) if len(old_avg_daily_velocity["time_unit"]) > 1 else 0,
            ]
    if len(glc_total_daily_velocity['Velocity_Category']) > 1:
        velocity_summary_data[
            f"Average Streak Duration (Hours) {glc_total_daily_velocity['Velocity_Category'][1]}"] = [
            round(
                glc_total_daily_velocity["daily_average_time_per_consecutive_group"][1],
                2,
            ),
            round(
                mid_total_daily_velocity["daily_average_time_per_consecutive_group"][1],
                2,
            ),
            round(
                old_total_daily_velocity["daily_average_time_per_consecutive_group"][1],
                2,
            )]



    gate_summary_data = {
        "Location": [
            glc_full_merged_df["gate"][0],
            mid_full_merged_df["gate"][0],
            old_full_merged_df["gate"][0],
        ],
        f"Average Daily {glc_avg_daily_gate['gate_status'][0]} Time (Hours) for gate": [
            round(glc_avg_daily_gate["time_unit"][0], 2),
            round(mid_avg_daily_gate["time_unit"][0], 2),
            round(old_avg_daily_gate["time_unit"][0], 2),
        ],
        f"Average {glc_total_daily_gate['gate_status'][0]} Duration (Hours) Per Streak": [
            round(
                glc_total_daily_gate["daily_average_time_per_consecutive_gate"][0], 2
            ),
            round(
                mid_total_daily_gate["daily_average_time_per_consecutive_gate"][0], 2
            ),
            round(
                old_total_daily_gate["daily_average_time_per_consecutive_gate"][0], 2
            ),
        ]}
    if len(glc_avg_daily_gate["gate_status"]) > 1:
        gate_summary_data[
            f"Average Daily {glc_avg_daily_gate['gate_status'][1]} Time (Hours) for gate"
        ] = [
            round(glc_avg_daily_gate["time_unit"][1], 2) if len(glc_avg_daily_gate["time_unit"]) > 1 else 0,
            round(mid_avg_daily_gate["time_unit"][1], 2) if len(mid_avg_daily_gate["time_unit"]) > 1 else 0,
            round(old_avg_daily_gate["time_unit"][1], 2) if len(old_avg_daily_gate["time_unit"]) > 1 else 0,
        ]   

    if len(glc_total_daily_gate['gate_status']) > 1:
        gate_summary_data[
            f"Average {glc_total_daily_gate['gate_status'][1]} Duration (Hours) Per Streak"] = [
            round(
                glc_total_daily_gate["daily_average_time_per_consecutive_gate"][1], 2
            ),
            round(
                mid_total_daily_gate["daily_average_time_per_consecutive_gate"][1], 2
            ),
            round(
                old_total_daily_gate["daily_average_time_per_consecutive_gate"][1], 2
            ),
        ]

    min_max_summary = {
        "Location": [
            location_gate[glc_full_merged_df["gate"][0]],
            location_gate[mid_full_merged_df["gate"][0]],
            location_gate[old_full_merged_df["gate"][0]],
        ],
        "Minimum velocity through fish passage (ft/s)": [
            round(min(glc_full_merged_df["velocity"]), 2),
            round(min(mid_full_merged_df["velocity"]), 2),
            round(min(old_full_merged_df["velocity"]), 2),
        ],
        "Maximum velocity through fish passage (ft/s)": [
            round(max(glc_full_merged_df["velocity"]), 2),
            round(max(mid_full_merged_df["velocity"]), 2),
            round(max(old_full_merged_df["velocity"]), 2),
        ],
    }

    # Create a DataFrame
    velocity_summary_df = pd.DataFrame(velocity_summary_data)
    gate_summary_df = pd.DataFrame(gate_summary_data)
    min_max_vel_summary_df = pd.DataFrame(min_max_summary)
    # if 'glc_full_merged_df' not in st.session_state:
    #     st.session_state.glc_full_merged_df = glc_full_merged_df
    glc_chart = generate_velocity_gate_charts(glc_full_merged_df)
    mid_chart = generate_velocity_gate_charts(mid_full_merged_df)
    old_chart = generate_velocity_gate_charts(old_full_merged_df, legend=True)
    #
    st.write("### Data Preview")
    data_preview_glc, data_preview_mid, data_preview_old = st.tabs(
        ["GLC", "MID", "OLD"]
    )
    data_preview_glc.dataframe(
        # glc_full_merged_df[["datetime", "node", "velocity", "unit"]]
        glc_full_merged_df
        .head(100)
        .style.format(precision=2)
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#4CAF50"),
                        ("color", "white"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [("background-color", "#f5f5f5")],
                },
            ]
        ),
        use_container_width=True,
    )

    data_preview_mid.dataframe(
        mid_full_merged_df.head(20)
        .style.format(precision=2)
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#4CAF50"),
                        ("color", "white"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [("background-color", "#f5f5f5")],
                },
            ]
        ),
        use_container_width=True,
    )

    data_preview_old.dataframe(
        old_full_merged_df.head(20)
        .style.format(precision=2)
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#4CAF50"),
                        ("color", "white"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [("background-color", "#f5f5f5")],
                },
            ]
        ),
        use_container_width=True,
    )
    # st.write("### Gate and Channel Locations")

    #     # shapefile_paths = [
    #     #     "data-raw/MSS_nodes/dsm2_nodes_newcs_extranodes.shp",
    #     #     "data-raw/fc2024.01_chan/FC2024.01_channels_centerlines.shp"
    #     # ]
    #     # nodes = gpd.read_file(shapefile_paths[0])
    #     # channels = gpd.read_file(shapefile_paths[1])
    #     # nodes_to_highlight = [112, 176, 69]
    #     # nodes_filter = nodes[nodes['id'].isin(nodes_to_highlight)]
    #     # channels_with_numbers = pd.read_csv('data-raw/channel_names_from_h5.csv')
    #     # channels_with_numbers = channels_with_numbers.rename(columns={'chan_no': 'id'})

    #     # channels_merge = pd.merge(
    #     #     channels,
    #     #     channels_with_numbers,
    #     #     how='left',
    #     #     left_on='id',
    #     #     right_on='id'
    #     # )

    #     # filtered_channels = channels_merge[channels_merge['id'].isin([211, 79, 134])]

    #     # Generate the map
    #     # gdfs, all_centroids= process_shapefiles(shapefile_paths)
    #     # nodes_filter, filtered_channels = transform_and_filter_geometries(nodes_filter, filtered_channels)
    #     # avg_lat, avg_lon = calculate_avg_lat_long(all_centroids)
    #     # map_object = create_multi_layer_map(gdfs=gdfs, avg_lat=avg_lat, avg_lon=avg_lon, filtered_gdf=nodes_filter, filtered_polylines=filtered_channels)
    #     # col1, col2, col3 = st.columns([1, 3, 1])
    #     # with col2:
    #     #     st_map = st_folium(map_object, width=1200, height=500)

    # # Display the selected data (If any)
    #     # st.dataframe(st.session_state.map_data)
    st.write("### Daily Gate Status Duration vs Daily Velocity Flow Duration")

    viz_1_tab1, viz_1_tab2 = st.tabs(["🗃 Data Summary", "📈 Chart"])
    # viz_1_tab1.write("### Data Summary")
    viz_1_tab1.write(f"##### {velocity_summary_stats_title}")
    viz_1_tab1.dataframe(
        velocity_summary_df.style.highlight_max(
            subset=velocity_summary_df.columns[1:], color="#ffffc5"
        ).format(precision=2)
    )
    viz_1_tab1.write("")
    viz_1_tab1.write(f"##### {min_max_summary_title}")
    viz_1_tab1.dataframe(
        min_max_vel_summary_df.style.highlight_max(
            subset=min_max_vel_summary_df.columns[1:], color="#ffffc5"
        ).format(precision=2)
    )
    viz_1_tab1.write("")
    viz_1_tab1.write(f"##### {gate_summary_stats_title}")
    viz_1_tab1.dataframe(
        gate_summary_df.style.highlight_max(
            subset=gate_summary_df.columns[1:], color="#ffffc5"
        ).format(precision=2)
    )
    # Altair Visualization
    # st.write('#')
    col1, col2, col3 = viz_1_tab2.columns([3, 3, 3], gap="small")
    with col1:
        st.altair_chart(glc_chart, use_container_width=True, theme=None)
        csv = convert_df(glc_full_merged_df)
        st.download_button(
            label="Download GLC Data",
            data=csv,
            file_name="glc_full_merged_df.csv",
            mime="text/csv",
        )
    with col2:
        st.altair_chart(mid_chart, use_container_width=True, theme=None)
        csv = convert_df(mid_full_merged_df)
        st.download_button(
            label="Download MID Data",
            data=csv,
            file_name="mid_full_merged_df.csv",
            mime="text/csv",
        )
    with col3:
        st.altair_chart(old_chart, use_container_width=True, theme=None)
        csv = convert_df(old_full_merged_df)
        st.download_button(
            label="Download OLD Data",
            data=csv,
            file_name="old_full_merged_df.csv",
            mime="text/csv",
        )
    st.write("###")
    st.write("### Flow Velocity and Gate Status Zoomed")
    default_start_date = pd.to_datetime(glc_full_merged_df["date"].min())
    default_end_date = pd.to_datetime(
        glc_full_merged_df["date"].min()
    ) + datetime.timedelta(days=7)

    start_date = default_start_date
    end_date = default_end_date

    if "selected_date_range" not in st.session_state:
        st.session_state.selected_date_range = None

    def submit_date_range():
        st.session_state.selected_date_range = st.session_state.date_range_input

    st.date_input(
        "Pick a date range:",
        value=(default_start_date, default_end_date),
        key="date_range_input",
        help="Select start and end dates",
    )
    st.button("Filter Data", on_click=submit_date_range)
    if st.session_state.selected_date_range:
        start_date, end_date = st.session_state.selected_date_range
        if start_date and end_date:
            st.write(f"Date Filtered.")

    start_date = str(start_date)
    end_date = str(end_date)

    filtered_glc_df = glc_full_merged_df[
        (glc_full_merged_df["date"] >= start_date)
        & (glc_full_merged_df["date"] <= end_date)
    ]
    filtered_mid_df = mid_full_merged_df[
        (mid_full_merged_df["date"] >= start_date)
        & (mid_full_merged_df["date"] <= end_date)
    ]
    filtered_old_df = old_full_merged_df[
        (old_full_merged_df["date"] >= start_date)
        & (old_full_merged_df["date"] <= end_date)
    ]

    filtered_glc_hydro_df = glc_hydro_df[
        (glc_hydro_df["datetime"] >= start_date)
        & (glc_hydro_df["datetime"] <= end_date)
    ]
    filtered_mid_hydro_df = mid_hydro_df[
        (mid_hydro_df["datetime"] >= start_date)
        & (mid_hydro_df["datetime"] <= end_date)
    ]
    filtered_old_hydro_df = old_hydro_df[
        (old_hydro_df["datetime"] >= start_date)
        & (old_hydro_df["datetime"] <= end_date)
    ]
    # print(filtered_glc_df)
    # #-------------------------------------------------------------------------------------------------------
    summary_stats_title = f"Summary stats from {start_date} to {end_date}."
    filtered_glc_avg_daily_velocity = calc_avg_daily_vel(filtered_glc_df)
    filtered_glc_avg_daily_gate = calc_avg_daily_gate(filtered_glc_df)

    filtered_old_avg_daily_velocity = calc_avg_daily_vel(filtered_old_df)
    filtered_old_avg_daily_gate = calc_avg_daily_gate(filtered_old_df)

    filtered_mid_avg_daily_velocity = calc_avg_daily_vel(filtered_mid_df)
    filtered_mid_avg_daily_gate = calc_avg_daily_gate(filtered_mid_df)

    weekly_min_date = min(filtered_glc_df["date"])
    weekly_max_date = max(filtered_glc_df["date"])


    weekly_summary_data = {
        "Location": [
            location_gate[glc_full_merged_df["gate"][0]],
            location_gate[mid_full_merged_df["gate"][0]],
            location_gate[old_full_merged_df["gate"][0]],
        ],
        f"Average Daily Time (Hours) {filtered_glc_avg_daily_velocity['Velocity_Category'][0]}": [
            round(filtered_glc_avg_daily_velocity["time_unit"][0], 2),
            round(filtered_mid_avg_daily_velocity["time_unit"][0], 2),
            round(filtered_old_avg_daily_velocity["time_unit"][0], 2),
        ],
        f"Average Daily {filtered_glc_avg_daily_gate['gate_status'][0]} Time (Hours) for gate": [
            round(filtered_glc_avg_daily_gate["time_unit"][0], 2),
            round(filtered_mid_avg_daily_gate["time_unit"][0], 2),
            round(filtered_old_avg_daily_gate["time_unit"][0], 2),
        ]
        }
    if len(filtered_glc_avg_daily_velocity["Velocity_Category"]) > 1:
        weekly_summary_data[
                f"Average Daily Time (Hours) {filtered_glc_avg_daily_velocity['Velocity_Category'][1]}"
            ] = [
                round(filtered_glc_avg_daily_velocity["time_unit"][1], 2) if len(filtered_glc_avg_daily_velocity["time_unit"]) > 1 else 0,
                round(filtered_mid_avg_daily_velocity["time_unit"][1], 2) if len(filtered_mid_avg_daily_velocity["time_unit"]) > 1 else 0,
                round(filtered_old_avg_daily_velocity["time_unit"][1], 2) if len(filtered_old_avg_daily_velocity["time_unit"]) > 1 else 0,
            ]
    if len(filtered_glc_avg_daily_gate["gate_status"]) > 1:
        weekly_summary_data[
            f"Average Daily {filtered_glc_avg_daily_gate['gate_status'][1]} Time (Hours) for gate"
        ] = [
            round(filtered_glc_avg_daily_gate["time_unit"][1], 2) if len(filtered_glc_avg_daily_gate["time_unit"]) > 1 else 0,
            round(filtered_mid_avg_daily_gate["time_unit"][1], 2) if len(filtered_mid_avg_daily_gate["time_unit"]) > 1 else 0,
            round(filtered_old_avg_daily_gate["time_unit"][1], 2) if len(filtered_old_avg_daily_gate["time_unit"]) > 1 else 0,
        ]
    # Create a DataFrame
    viz_2_tab1, viz_2_tab2 = st.tabs(["🗃 Data Summary", "📈 Chart"])
    weekly_summary_df = pd.DataFrame(weekly_summary_data)
    viz_2_tab1.write(f"##### {summary_stats_title}")
    viz_2_tab1.dataframe(
        weekly_summary_df.style.highlight_max(
            subset=weekly_summary_df.columns[1:], color="#ffffc5"
        ).format(precision=2)
    )
    # #-------------------------------------------------------------------------------------------------------
    glc_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_glc_df)
    mid_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_mid_df)
    old_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_old_df)
    # st.write("#")
    col1, col2, col3 = viz_2_tab2.columns([10, 10, 10], gap="small")
    with col1:
        st.altair_chart(
            glc_zoomed_vel_chart,
            # use_container_width=True,
            theme=None,
        )
        # st.altair_chart(glc_zoomed_vel_chart[1], use_container_width=True, theme=None)
    with col2:
        st.altair_chart(
            mid_zoomed_vel_chart,
            # use_container_width=True,
            theme=None,
        )
    with col3:
        st.altair_chart(
            old_zoomed_vel_chart,
            # use_container_width=True,
            theme=None,
        )
    glc_zoomed_hydro_chart = generate_water_level_chart(
        filtered_glc_hydro_df, filtered_glc_df
    )
    mid_zoomed_hydro_chart = generate_water_level_chart(
        filtered_mid_hydro_df, filtered_mid_df
    )
    old_zoomed_hydro_chart = generate_water_level_chart(
        filtered_old_hydro_df, filtered_old_df
    )
    # st.write("#")
    col1, col2, col3 = viz_2_tab2.columns([3, 3, 3], gap="small")
    with col1:
        st.altair_chart(glc_zoomed_hydro_chart, use_container_width=True, theme=None)
        csv = convert_df(glc_hydro_df)
        st.download_button(
            label="Download GLC Hydro Data",
            data=csv,
            file_name="glc_hydro_df.csv",
            mime="text/csv",
        )
    with col2:
        st.altair_chart(mid_zoomed_hydro_chart, use_container_width=True, theme=None)
        csv = convert_df(mid_hydro_df)
        st.download_button(
            label="Download MID Hydro Data",
            data=csv,
            file_name="mid_hydro_df.csv",
            mime="text/csv",
        )
    with col3:
        st.altair_chart(old_zoomed_hydro_chart, use_container_width=True, theme=None)
        csv = convert_df(old_hydro_df)
        st.download_button(
            label="Download OLD Hydro Data",
            data=csv,
            file_name="OLD_hydro_df.csv",
            mime="text/csv",
        )
else:
    st.write(f"Year {selected_year} did not return any data, contact app admin.")

    # st.session_state["glc_full_merged_df"] = age
