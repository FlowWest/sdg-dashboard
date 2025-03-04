import folium
import streamlit as st
from streamlit_folium import folium_static, st_folium
import geopandas as gpd
import pandas as pd
import branca.colormap as cm
import numpy as np


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


baseline_data = pd.read_csv("baseline-data.csv", parse_dates=["Date"])
scenario_data = pd.read_csv("fpv2mb-data.csv", parse_dates=["Date"])


baseline_data["year"] = baseline_data["Date"].dt.year
baseline_data["month"] = baseline_data["Date"].dt.month
baseline_by_month = (
    baseline_data.groupby(["channel_id", "month"])["daily_avg"].mean().reset_index()
)
may_baseline = baseline_by_month[baseline_by_month["month"] == 5]

dsm2_channels_with_stage = dsm2_channels.merge(
    may_baseline, how="left", left_on="id", right_on="channel_id"
)

st.title("Map")

col_left, col_right = st.columns([1, 4])

linear = cm.LinearColormap(
    colors=["blue", "white", "red"],
    vmin=may_baseline["daily_avg"].min(),
    vmax=may_baseline["daily_avg"].max(),
)


def get_or_create_map():
    if "map" not in st.session_state or st.session_state is None:
        m = folium.Map(
            location=[
                dsm2_channels_with_stage.union_all().centroid.y,
                dsm2_channels_with_stage.union_all().centroid.x,
            ],
            width="100%",
            height="100%",
        )
        m.add_child(linear)
        channel_lines = dsm2_channels_with_stage[
            dsm2_channels_with_stage.geometry.type == "LineString"
        ]
        for index, row in channel_lines.iterrows():
            locs = [(y, x) for x, y in row.geometry.coords]
            folium.PolyLine(
                locations=locs,
                color="#808080"
                if np.isnan(row["daily_avg"])
                else linear(row["daily_avg"]),
                weight=5,
            ).add_to(m)
        st.session_state.map = m
    return st.session_state.map


with col_left:
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
    st.select_slider("Select Year", options=range(2016, 2024))
    filtered_data = baseline_data[baseline_data["year"] == selected_year]

with col_right:
    folium_static(get_or_create_map(), width=1400, height=600)
