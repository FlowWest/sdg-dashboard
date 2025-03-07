import folium
import streamlit as st
from streamlit_folium import folium_static, st_folium
import geopandas as gpd
import pandas as pd
import branca.colormap as cm
import numpy as np
from db import get_dsm2_daily_summaries, get_dsm2_daily_summaries


dsm2_channels = gpd.read_file(
    "data-raw/fc2024.01_chan/FC2024.01_channels_centerlines.shp"
)
dsm2_channels = dsm2_channels.to_crs(epsg=4326)
month_names = {
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
}

col_left, col_right = st.columns([1, 4])


@st.cache_data
def get_dsm2_daily_summaries_for_map(scenario_id, years):
    return get_dsm2_daily_summaries(scenario_id, years)


def get_or_create_map(map_data, color_map):
    if "map" not in st.session_state:
        m = folium.Map(
            location=[
                map_data.union_all().centroid.y,
                map_data.union_all().centroid.x,
            ],
            width="100%",
            height="100%",
        )
        m.add_child(color_map)
        channel_lines = map_data[map_data.geometry.type == "LineString"]
        for index, row in channel_lines.iterrows():
            locs = [(y, x) for x, y in row.geometry.coords]
            folium.PolyLine(
                locations=locs,
                color="#808080"
                if np.isnan(row["daily_minimum"])
                else color_map(row["daily_minimum"]),
                weight=5,
                tooltip=folium.Tooltip(f"Value: {row.get("daily_minimum")}"),
            ).add_to(m)
        st.session_state.map = m
    return st.session_state.map


# set up UI stuff
with col_left:
    st.title("Map")
    st.markdown(
        """
        Each difference (min, max, mean) 
        represents a comparison of the summarized values for a specific year, 
        month with an option to summarize by month or day. 
        Daily summaries are the summary of daily 5-minute increments. 
        Monthly summarize represent the min, max, or mean of the daily summaries.
        """
    )

    selected_year = st.selectbox("Select Year", options=range(2016, 2024))
    st.selectbox("Select Month", options=month_names)

    left_selected_scenario = st.selectbox(
        "Select Scenario A",
        options=["FPV2Ma", "FPV2Mb", "baseline"],
        key="left_selected_scenario",
    )

    right_selected_scenario = st.selectbox(
        "Select Scenario B",
        options=["FPV2Ma", "FPV2Mb", "baseline"],
        key="right_seleted_scenario",
    )

    map_options = st.popover("Additional Plot Options")
    selected_map_color_ramp = map_options.selectbox("hello", options=[1, 2, 2, 3])


monthly_summaries = get_dsm2_daily_summaries_for_map(
    scenario_id=12, years=[selected_year]
)
monthly_summaries["month"] = monthly_summaries["date"].dt.month
monthly_summaries = monthly_summaries[monthly_summaries["month"] == 6]
monthly_summaries_by_month = (
    monthly_summaries.groupby(["channel_id", "month"])["daily_minimum"]
    .mean()
    .reset_index()
)


dsm2_channels_with_stage = dsm2_channels.merge(
    monthly_summaries_by_month, how="left", left_on="id", right_on="channel_id"
)

linear_map = cm.LinearColormap(
    colors=["blue", "white", "red"],
    vmin=dsm2_channels_with_stage["daily_minimum"].min(),  # type: ignore
    vmax=dsm2_channels_with_stage["daily_minimum"].max(),  # type: ignore
)

# becuase my laptop crashes
if len(dsm2_channels_with_stage) > 600:
    raise ValueError(
        f"too much data, saving you from having to hard reboot your laptop, the left is {len(dsm2_channels)} and right is {len(monthly_summaries)}"
    )

with col_right:
    folium_static(
        get_or_create_map(dsm2_channels_with_stage, linear_map), width=1400, height=600
    )
    st.dataframe(dsm2_channels_with_stage)
