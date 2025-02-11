import pandas as pd
import altair as alt
from pandas import DataFrame
import numpy as np
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium


location_gate = {
    "glc": "Grantline",
    "mid": "MiddleRiver",
    "old": "OldRiver",
}


def post_process_gateop(
    gate_data, model, gate, year=None, start_year=None, end_year=None
):
    # gate_data = multi_model_data[model][gate]['gate_operation_data']
    if year:
        gate_data["year"] = gate_data["datetime"].dt.year
        gate_data = gate_data[gate_data["year"] == year]
    if start_year:
        gate_data["date"] = gate_data["datetime"].dt.year
        gate_data = gate_data[
            (gate_data["date"] >= start_year) & (gate_data["date"] <= end_year)
        ]
    gate_data = gate_data.loc[gate_data["datetime"].dt.month.between(5, 11)]
    # gate_data['datetime'].dt.month.between(5, 11)
    # ]
    gate_up_df = gate_data[["datetime", "value"]]
    gate_up_df["gate_status"] = gate_up_df["value"] >= 10
    gate_up_df["consecutive_groups"] = (
        gate_up_df["value"] != gate_up_df["value"].shift()
    ).cumsum()
    gate_up_df["min_datetime"] = gate_up_df.groupby("consecutive_groups")[
        "datetime"
    ].transform("min")
    gate_up_df["max_datetime"] = gate_up_df.groupby("consecutive_groups")[
        "datetime"
    ].transform("max")
    consecutive_streaks = (
        gate_up_df.groupby(
            ["consecutive_groups", "value", "min_datetime", "max_datetime"]
        )
        .size()
        .reset_index(name="count")
    )
    consecutive_streaks["streak_duration"] = consecutive_streaks["count"] * 15 / 60
    consecutive_streaks_clean = consecutive_streaks.drop(
        ["value", "consecutive_groups", "max_datetime"], axis=1
    )
    # print(consecutive_streaks_clean.head())
    merged_gate_df = pd.merge(
        gate_up_df,
        consecutive_streaks_clean,
        left_on="min_datetime",
        right_on="min_datetime",
    )
    merged_gate_df = merged_gate_df.drop(["consecutive_groups", "value"], axis=1)
    merged_gate_df = merged_gate_df.rename(
        columns={
            "min_datetime": "gate_min_datetime",
            "max_datetime": "gate_max_datetime",
            "count": "gate_count",
            "streak_duration": "gate_streak_duration",
        }
    )
    return merged_gate_df


def post_process_velocity(data, model, gate, year=None, start_year=None, end_year=None):
    vel_zoom_df = data
    if year:
        vel_zoom_df["year"] = vel_zoom_df["datetime"].dt.year
        vel_zoom_df = vel_zoom_df[vel_zoom_df["year"] == year]
    vel_zoom_df = vel_zoom_df.loc[vel_zoom_df["datetime"].dt.month.between(5, 11)]
    if start_year:
        vel_zoom_df["year"] = vel_zoom_df["datetime"].dt.year
        vel_zoom_df = vel_zoom_df[
            (vel_zoom_df["year"] >= start_year) & (vel_zoom_df["year"] <= end_year)
        ]
    vel_zoom_df["Velocity_Category"] = np.where(
        vel_zoom_df["value"] >= 8, "Over 8ft/s", "Under 8ft/s"
    )
    # .shift shift value down and compare each value with the previous row; increase value when rows are different
    vel_zoom_df["consecutive_groups"] = (
        vel_zoom_df["Velocity_Category"] != vel_zoom_df["Velocity_Category"].shift()
    ).cumsum()
    vel_zoom_df["min_datetime"] = vel_zoom_df.groupby("consecutive_groups")[
        "datetime"
    ].transform("min")
    vel_zoom_df["max_datetime"] = vel_zoom_df.groupby("consecutive_groups")[
        "datetime"
    ].transform("max")
    vel_zoom_df["date"] = vel_zoom_df["datetime"].dt.date.astype(str)
    consecutive_streaks_vel = (
        vel_zoom_df.groupby(
            ["consecutive_groups", "Velocity_Category", "min_datetime", "max_datetime"]
        )
        .size()
        .reset_index(name="count")
    )
    consecutive_streaks_vel["streak_duration"] = (
        consecutive_streaks_vel["count"] * 15 / 60
    )

    consecutive_streaks_vel_clean = consecutive_streaks_vel.drop(
        ["consecutive_groups", "Velocity_Category", "max_datetime"], axis=1
    )
    merged_vel_df = pd.merge(
        vel_zoom_df,
        consecutive_streaks_vel_clean,
        left_on="min_datetime",
        right_on="min_datetime",
    )

    return merged_vel_df


def post_process_full_data(
    gate_data, flow_data, model, gate, year=None, start_year=None, end_year=None
):
    merged_gate_df = post_process_gateop(
        gate_data, model, gate, year, start_year, end_year
    )
    merged_vel_df = post_process_velocity(
        flow_data, model, gate, year, start_year, end_year
    )
    full_merged_df = pd.merge(
        merged_vel_df, merged_gate_df, left_on="datetime", right_on="datetime"
    )
    full_merged_df["time_unit"] = 0.25

    full_merged_df["gate_status"] = np.where(
        full_merged_df["gate_status"], "Closed", "Open"
    )
    full_merged_df["week"] = full_merged_df["datetime"].dt.isocalendar().week
    full_merged_df["gate"] = gate
    full_merged_df["model"] = model

    return full_merged_df


def post_process_hydro_data(
    data, model, gate, year=None, start_date=None, end_date=None
):
    hydro_df = data
    if year:
        hydro_df["year"] = hydro_df["datetime"].dt.year
        hydro_df = hydro_df[hydro_df["year"] == year]
    hydro_df = hydro_df.loc[hydro_df["datetime"].dt.month.between(5, 11)]
    hydro_df["time_unit"] = 0.25
    hydro_df = hydro_df.rename(columns={"value": "water_level"})
    hydro_df["week"] = hydro_df["datetime"].dt.isocalendar().week
    return hydro_df


# @st.cache_data
def calc_avg_daily_vel(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of total amount of time velocity is above and below 8ft/s.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """

    daily_velocity = (
        post_processed_data.groupby(["date", "Velocity_Category"])["time_unit"]
        .sum()
        .reset_index()
    )
    avg_daily_velocity = pd.DataFrame(
        daily_velocity.groupby("Velocity_Category")["time_unit"].sum()
        / daily_velocity["date"].nunique()
    ).reset_index()

    return avg_daily_velocity


# @st.cache_data
def calc_avg_daily_gate(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of total amount of time gate is open and closed.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """

    daily_gate = (
        post_processed_data.groupby(["date", "gate_status"])["time_unit"]
        .sum()
        .reset_index()
    )
    avg_daily_gate = pd.DataFrame(
        daily_gate.groupby("gate_status")["time_unit"].sum()
        / daily_gate["date"].nunique()
    ).reset_index()
    return avg_daily_gate


# @st.cache_data
def calc_avg_len_consec_vel(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of length of consecutive hours velocity is above and below 8ft/s.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """

    daily_velocity_stats = (
        post_processed_data.groupby(["date", "Velocity_Category"])
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
    daily_average_per_duration_per_velocity_over_period = (
        daily_velocity_stats.groupby(["Velocity_Category"])[
            "daily_average_time_per_consecutive_group"
        ]
        .mean()
        .reset_index()
    )

    return daily_average_per_duration_per_velocity_over_period


# @st.cache_data
def calc_avg_len_consec_gate(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of length of consecutive hours gate is open or closed.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """

    daily_gate_stats = (
        post_processed_data.groupby(["date", "gate_status"])
        .agg(
            unique_gate_count=("gate_count", "nunique"), total_time=("time_unit", "sum")
        )
        .reset_index()
    )

    daily_gate_stats["daily_average_time_per_consecutive_gate"] = (
        daily_gate_stats["total_time"] / daily_gate_stats["unique_gate_count"]
    )
    daily_average_per_duration_per_gate_over_period = (
        daily_gate_stats.groupby(["gate_status"])[
            "daily_average_time_per_consecutive_gate"
        ]
        .mean()
        .reset_index()
    )

    return daily_average_per_duration_per_gate_over_period


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


# @st.cache_data
def generate_velocity_gate_charts(full_merged_df, legend=None):
    """
    Generate bar charts for daily velocity and gate status summaries.

    Parameters:
    full_merged_df (pd.DataFrame): DataFrame containing the merged data with the necessary columns.

    Returns:
    alt.Chart: Combined Altair charts for velocity and gate status duration summaries.
    """
    # Compute summary statistics
    summary_stats_vel = (
        full_merged_df.groupby(["week", "date", "Velocity_Category"]).agg(
            total_velocity_duration=("time_unit", "sum")
        )
    ).reset_index()

    summary_stats_dgl = (
        full_merged_df.groupby(["week", "date", "gate_status"]).agg(
            total_gate_duration=("time_unit", "sum")
        )
    ).reset_index()

    # Extract unique gate identifier
    gate = full_merged_df["gate"].unique()[0]
    model = full_merged_df["model"].unique()[0]

    # Define color palette
    color_palette = {
        "Velocity_Category": {
            "Over 8ft/s": "#a6cee3",
            "Under 8ft/s": "#1f78b4",
        },  # Blues
        "gate_status": {"Closed": "#b2df8a", "Open": "#33a02c"},  # Greens
    }

    # Define brush for selection
    brush = alt.selection_interval(
        encodings=["x"], mark=alt.BrushConfig(stroke="cyan", strokeOpacity=1)
    )

    # Velocity bar chart
    if legend:
        base_vel = (
            alt.Chart(summary_stats_vel, width=625, height=300)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("total_velocity_duration:Q", title="Hours"),
                color=alt.condition(
                    brush,
                    alt.Color(
                        "Velocity_Category:N",
                        title="Velocity",
                        scale=alt.Scale(
                            domain=color_palette["Velocity_Category"].keys(),
                            range=color_palette["Velocity_Category"].values(),
                        ),
                    ),
                    alt.value("lightgray"),  # Gray for unselected
                ),
                tooltip=["date:T", "Velocity_Category:N", "total_velocity_duration:Q"],
            )
            .properties(
                title=f"Daily Velocity at {location_gate[gate]} Over/Under 8 ft/s Duration Summary"
            )
        )
    else:
        base_vel = (
            alt.Chart(summary_stats_vel, width=550, height=300)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("total_velocity_duration:Q", title="Hours"),
                color=alt.condition(
                    brush,
                    alt.Color(
                        "Velocity_Category:N",
                        title="Velocity Category",
                        scale=alt.Scale(
                            domain=color_palette["Velocity_Category"].keys(),
                            range=color_palette["Velocity_Category"].values(),
                        ),
                        legend=None,
                    ),
                    alt.value("lightgray"),  # Gray for unselected
                ),
                tooltip=["date:T", "Velocity_Category:N", "total_velocity_duration:Q"],
            )
            .properties(
                title=f"Daily Velocity at {location_gate[gate]} Over/Under 8 ft/s Duration Summary"
            )
        )

    upper_vel = base_vel.mark_bar(
        width=alt.RelativeBandSize(0.7), stroke="grey", strokeWidth=0.5
    ).encode(alt.X("date:T", title="Date").scale(domain=brush))

    lower_vel = base_vel.properties(height=90).add_params(brush)

    vel_bar_chart = upper_vel & lower_vel

    # Gate bar chart
    if legend:
        base_gate = (
            alt.Chart(summary_stats_dgl, width=625, height=300)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("total_gate_duration:Q", title="Hours"),
                color=alt.condition(
                    brush,
                    alt.Color(
                        "gate_status:N",
                        title="Gate Status",
                        scale=alt.Scale(
                            domain=color_palette["gate_status"].keys(),
                            range=color_palette["gate_status"].values(),
                        ),
                    ),
                    alt.value("lightgray"),  # Gray for unselected
                ),
                tooltip=["date:T", "gate_status:N", "total_gate_duration:Q"],
            )
            .properties(title=f"Daily {gate} Gate Status Duration Summary")
        )
    else:
        base_gate = (
            alt.Chart(summary_stats_dgl, width=550, height=300)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("total_gate_duration:Q", title="Hours"),
                color=alt.condition(
                    brush,
                    alt.Color(
                        "gate_status:N",
                        title="Gate Status",
                        scale=alt.Scale(
                            domain=color_palette["gate_status"].keys(),
                            range=color_palette["gate_status"].values(),
                        ),
                        legend=None,
                    ),
                    alt.value("lightgray"),  # Gray for unselected
                ),
                tooltip=["date:T", "gate_status:N", "total_gate_duration:Q"],
            )
            .properties(title=f"Daily {gate} Gate Status Duration Summary")
        )

    upper_gate = base_gate.mark_bar(
        width=alt.RelativeBandSize(0.7), stroke="grey", strokeWidth=0.5
    ).encode(alt.X("date:T", title="Date").scale(domain=brush))

    lower_gate = base_gate.properties(height=90).add_params(brush)

    gate_bar_chart = upper_gate & lower_gate

    # Combine charts
    combined_bar_charts = (
        alt.vconcat(gate_bar_chart, vel_bar_chart)
        .resolve_scale(color="independent")
        .properties(title=f"Model: {model}")
        .configure_legend(
            orient="right",
            offset=5,
        )
    )

    return combined_bar_charts


# @st.cache_data
def generate_zoomed_velocity_charts(filtered_merged_df):
    color_palette = {
        "Velocity_Category": {
            "Over 8ft/s": "#a6cee3",
            "Under 8ft/s": "#1f78b4",
        },  # Blues
        "gate_status": {"Closed": "#b2df8a", "Open": "#33a02c"},  # Greens
    }
    gate = filtered_merged_df["gate"].unique()[0]
    model = filtered_merged_df["model"].unique()[0]
    shared_y_scale = alt.Scale(domain=[-5, 17])

    interval = alt.selection_interval(
        encodings=["x"], mark=alt.BrushConfig(fill="blue")
    )
    base = (
        alt.Chart(filtered_merged_df)
        .mark_line(color=color_palette["Velocity_Category"]["Under 8ft/s"])
        .encode(
            x=alt.X(
                "yearmonthdatehoursminutes(datetime):T",
                title="Datetime",
                axis=alt.Axis(format="%b %d, %Y", labelAngle=-45, title="Date"),
            ),
            y=alt.Y("velocity:Q", title="Velocity (ft/s)", scale=shared_y_scale),
        )
        .add_params(interval)
        .properties(
            title=f"Fish Passage Velocity and {gate} Gate Status Zoomed", height=300
        )
    )

    closed_gates = (
        filtered_merged_df[["gate_min_datetime", "gate_max_datetime", "gate_status"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    color_scale = alt.Scale(
        domain=["Closed"], range=[color_palette["gate_status"]["Closed"]]
    )
    area_gate_true = (
        alt.Chart(closed_gates)
        .mark_rect(color="orange")
        .encode(
            x="gate_min_datetime:T",
            x2="gate_max_datetime:T",
            opacity=alt.value(0.2),
            color=alt.Color(
                "gate_status:N",
                scale=color_scale,
                legend=alt.Legend(title="Gate Status"),
            ),
            # color=alt.value(color_palette["gate_status"]["Closed"]),
            # color=alt.condition(interval, alt.value('orange'), alt.value('lightgray'))
        )
        .transform_filter(alt.datum.gate_status == "Closed")
    )
    yrule = (
        alt.Chart(filtered_merged_df)
        .mark_rule(color="red", strokeDash=[12, 6], size=1.5)
        .encode(y=alt.datum(8))
        .properties(width=300, height=300)
    )

    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["datetime"], empty=False
    )
    points = base.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # # Draw a rule at the location of the selection
    rules = (
        alt.Chart(filtered_merged_df)
        .transform_calculate(
            FlowVelocityDuration="'Flow ' + datum.Velocity_Category + ' duration is ' + datum.streak_duration + ' hours'",
            GateStatusDuration="'Gate ' + datum.gate_status + ' duration is ' + datum.gate_streak_duration + ' hours'",
        )
        .mark_rule(color="gray")
        .encode(
            x="datetime:T",
            opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("yearmonthdatehoursminutes(datetime):T", title="Datetime"),
                alt.Tooltip("velocity:Q", title="Velocity (ft/s)", format=".2f"),
                alt.Tooltip("FlowVelocityDuration:N", title="Flow Velocity Duration"),
                alt.Tooltip("GateStatusDuration:N", title="Gate Status Duration"),
            ],
        )
        .add_params(nearest)
    )

    vel_text = (
        alt.Chart(filtered_merged_df)
        .mark_text(align="right")
        .encode(y=alt.Y("stat:N", axis=None), text=alt.Text("summary:N"))
        .transform_filter(interval)
        .transform_aggregate(
            max_velocity="max(velocity)",
            min_velocity="min(velocity)",
            avg_velocity="mean(velocity)",
        )
        .transform_fold(
            ["max_velocity", "min_velocity", "avg_velocity"],  # Separate each statistic
            as_=["stat", "value"],
        )
        .transform_calculate(summary='datum.stat + ": " + format(datum.value, ".2f")')
    )
    velocity = vel_text.encode(text="summary:N").properties(
        title=alt.Title(text="Flow Velocity Summary", align="center")
    )

    vel_duration_text = (
        alt.Chart(filtered_merged_df)
        .mark_text(align="right")
        .encode(y=alt.Y("Velocity_Category:N", axis=None), text=alt.Text("summary:N"))
        .transform_filter(interval)
        .transform_aggregate(total_time="sum(time_unit)", groupby=["Velocity_Category"])
        .transform_fold(
            ["total_time"],  # Separate each statistic
            as_=["stat", "value"],
        )
        .transform_calculate(
            summary='datum.Velocity_Category + ": " + format(datum.total_time, ".2f") + " hours"'
        )
    )
    velocity_duration = vel_duration_text.encode(text="summary:N").properties(
        title=alt.Title(text="Velocity Duration Summary", align="center")
    )

    gate_duration_text = (
        alt.Chart(filtered_merged_df)
        .mark_text(align="right")
        .encode(y=alt.Y("gate_status:N", axis=None), text=alt.Text("summary:N"))
        .transform_filter(interval)
        .transform_aggregate(total_time="sum(time_unit)", groupby=["gate_status"])
        .transform_fold(
            ["total_time"],  # Separate each statistic
            as_=["stat", "value"],
        )
        .transform_calculate(
            summary='datum.gate_status + ": " + format(datum.total_time, ".2f") + " hours"'
        )
    )
    gate_duration = gate_duration_text.encode(text="summary:N").properties(
        title=alt.Title(text="Gate Status Duration Summary", align="center")
    )

    layered_chart = alt.layer(base, points, yrule, rules, area_gate_true).properties(
        width=500, height=400
    )
    velocity = velocity.properties(width=200, height=100)
    # daily_velocity = daily_velocity.properties(width=200, height=100)
    velocity_duration = velocity_duration.properties(width=200, height=100)
    gate_duration = gate_duration.properties(width=200, height=100)

    horizontal_text_summary = alt.hconcat(velocity, velocity_duration)
    text_summary = alt.vconcat(horizontal_text_summary, gate_duration)

    combined_chart = (
        alt.vconcat(
            layered_chart,
            # velocity
            text_summary,
        )
        .properties(title=f"Model: {model}")
        .configure_legend(
            orient="right",
            offset=10,
        )
    )

    # return (layered_chart, text_summary)
    return combined_chart


# @st.cache_data
def process_shapefiles(shapefile_paths):
    gdfs = []

    for path in shapefile_paths:
        # Read shapefile
        all_centroids = []
        gdf = gpd.read_file(path)

        # Ensure the CRS is EPSG:4326
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Remove rows with invalid or missing geometries
        gdf = gdf[~gdf.geometry.isna()]
        nodes_to_highlight = [112, 176, 69]
        gdf = gdf[gdf["id"].isin(nodes_to_highlight)]
        centroid = gdf.geometry.centroid
        center = [centroid.y.mean(), centroid.x.mean()]

        gdfs.append((gdf, center))
        all_centroids.append(center)

    return gdfs, all_centroids


def transform_and_filter_geometries(nodes_filter, filtered_channels):
    """
    Transforms the CRS of nodes and channels to EPSG:4326, removes invalid geometries

    Parameters:
        nodes_filter (GeoDataFrame): The GeoDataFrame containing node geometries.
        filtered_channels (GeoDataFrame): The GeoDataFrame containing channel geometries.

    Returns:
        tuple: Transformed nodes_filter (GeoDataFrame), transformed filtered_channels (GeoDataFrame),
               average latitude, average longitude.
    """
    # Transform nodes_filter CRS to EPSG:4326 if it's not already
    if nodes_filter.crs != "EPSG:4326":
        nodes_filter = nodes_filter.to_crs("EPSG:4326")
    nodes_filter = nodes_filter[
        ~nodes_filter.geometry.isna()
    ]  # Remove invalid geometries

    # Transform filtered_channels CRS to EPSG:4326 if it's not already
    if filtered_channels.crs != "EPSG:4326":
        filtered_channels = filtered_channels.to_crs("EPSG:4326")
    filtered_channels = filtered_channels[
        ~filtered_channels.geometry.isna()
    ]  # Remove invalid geometries

    return nodes_filter, filtered_channels


def calculate_avg_lat_long(all_centroids):
    avg_lat = sum([c[0] for c in all_centroids]) / len(all_centroids)
    avg_lon = sum([c[1] for c in all_centroids]) / len(all_centroids)
    return avg_lat, avg_lon


def create_multi_layer_map(
    gdfs, avg_lon, avg_lat, filtered_gdf=None, filtered_polylines=None
):
    # Create the Folium map
    if "map" not in st.session_state or st.session_state.map is None:
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)

        if filtered_gdf is not None:
            for _, row in filtered_gdf.iterrows():
                # Define the popup content for the markers
                popup_content = f"ID: {row['id']}"

                # Define tooltip content for the marker (appears on hover)
                tooltip_content = f"ID: {row['id']}"

                # Add the marker with a popup, icon, and tooltip
                folium.Marker(
                    location=[
                        row.geometry.y,
                        row.geometry.x,
                    ],  # Point geometry: extract lat/lon
                    popup=popup_content,  # Popup content
                    icon=folium.Icon(color="red", icon="circle"),  # Red marker icon
                    tooltip=tooltip_content,  # Tooltip on hover
                ).add_to(m)

        # Loop through the filtered polylines
        if filtered_polylines is not None:
            for _, row in filtered_polylines.iterrows():
                # Define tooltip content
                tooltip_content = f"""
                id: {row['id']}<br>
                name: {row.get('name', 'N/A')}<br>
                """

                # Define popup content
                popup_content = f"""
                id: {row['id']}<br>
                name: {row.get('name', 'N/A')}<br>
                distance: {row.get('distance', 'N/A')}<br>
                variable: {row.get('variable', 'N/A')}<br>
                interval: {row.get('interval', 'N/A')}<br>
                period_op: {row.get('period_op', 'N/A')}
                """

                # Extract coordinates for the LineString
                coordinates = [
                    (point[1], point[0]) for point in row.geometry.coords
                ]  # Convert (x, y) to (lat, lon)

                # Add the LineString to the map with a Tooltip
                folium.PolyLine(
                    locations=coordinates,
                    color="darkgreen",
                    tooltip=folium.Tooltip(tooltip_content),
                    popup=folium.Popup(popup_content, max_width=300),
                ).add_to(m)
        st.session_state.map = m

    return st.session_state.map


def generate_water_level_chart(filtered_hydro_df, filtered_merged_df):
    # @st.cache_data
    color_palette = {
        "Velocity_Category": {
            "Over 8ft/s": "#a6cee3",
            "Under 8ft/s": "#1f78b4",
        },  # Blues
        "gate_status": {"Closed": "#b2df8a", "Open": "#33a02c"},  # Greens
    }
    gate = filtered_merged_df["gate"].unique()[0]
    model = filtered_merged_df["model"].unique()[0]
    shared_y_scale = alt.Scale(domain=[0, 8])
    interval = alt.selection_interval(
        encodings=["x"], mark=alt.BrushConfig(fill="blue")
    )
    base_elevation = (
        alt.Chart(filtered_hydro_df)
        .mark_line()
        .encode(
            x=alt.X(
                "yearmonthdatehoursminutes(datetime):T",
                title="Date",
                axis=alt.Axis(format="%b %d, %Y", labelAngle=-45),
            ),
            y=alt.Y("water_level:Q", title="Feet", scale=shared_y_scale),
            # color='scenario:N'
        )
        .transform_fold(
            ["scenario"],  # Columns to be "melted" into a long format
            as_=["model", "value"],  # New column names
        )
        .add_params(interval)
        .properties(title=f"Summary of Minimum Stage At {gate}")
    )
    yrule_wl = (
        alt.Chart(filtered_hydro_df)
        .mark_rule(color="purple", strokeDash=[12, 6], size=1.5)
        .encode(y=alt.datum(2.3))
    )
    yrule_wl_text = (
        alt.Chart(filtered_hydro_df)
        .mark_text(
            text="Water Level Compliance",
            align="left",
            baseline="bottom",
            fontSize=12,
            color="grey",
            dx=5,  # Offset text slightly to the right of the rule
        )
        .encode(
            y=alt.datum(2.3)  # Same y position as the rule
        )
    )
    nearest_elev = alt.selection_point(
        nearest=True, on="pointerover", fields=["datetime"], empty=False
    )
    points_elev = base_elevation.mark_point().encode(
        opacity=alt.condition(nearest_elev, alt.value(1), alt.value(0))
    )
    rules_elev = (
        alt.Chart(filtered_hydro_df)
        .mark_rule(color="gray", opacity=0)
        .encode(
            x="datetime:T",
            opacity=alt.condition(nearest_elev, alt.value(0.3), alt.value(0)),
        )
        .add_params(nearest_elev)
    )
    when_near = alt.when(nearest_elev)
    text = (
        base_elevation.mark_text(align="left", dx=5, dy=-5)
        .transform_calculate(label='format(datum.water_level, ".2f") + " feet"')
        .encode(text=when_near.then("label:N").otherwise(alt.value(" ")))
    )
    average_scenario_stage = (
        alt.Chart(filtered_hydro_df)
        .mark_text(align="right")
        .encode(y=alt.Y("stat:N", axis=None), text=alt.Text("summary:N"))
        .transform_filter(interval)
        .transform_aggregate(average_stage="mean(water_level)")
        .transform_fold(
            ["average_stage"],  # Separate each statistic
            as_=["stat", "value"],
        )
        .transform_calculate(summary='format(datum.average_stage, ".2f") + " feet"')
    )
    avg_stage = average_scenario_stage.encode(text="summary:N").properties(
        title=alt.Title(text=f"Average {model} Minimum Stage", align="center")
    )
    scenario_duration_below_wl = (
        alt.Chart(filtered_hydro_df)
        .mark_text(align="right")
        .encode(
            y=alt.Y("wl_stat:N", axis=None),
            text=alt.Text("below_wl:N"),
            # ).transform_filter(
            #     interval
        )
        .transform_filter("datum.FPV1Ma < 2.3")
        .transform_aggregate(total_time_below_wl="sum(time_unit)")
        .transform_calculate(
            below_wl='format(datum.total_time_below_wl, ".2f") + " hour"'
        )
        .properties(
            title=alt.Title(
                text="Scenario Below Water Level Compliance", align="center"
            )
        )
    )
    closed_gates = (
        filtered_merged_df[["gate_min_datetime", "gate_max_datetime", "gate_status"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    area_gate_true = (
        alt.Chart(closed_gates)
        .mark_rect(color="orange")
        .encode(
            x="gate_min_datetime:T",
            x2="gate_max_datetime:T",
            opacity=alt.value(0.2),
            color=alt.value(color_palette["gate_status"]["Closed"]),
            # color=alt.condition(interval, alt.value('orange'), alt.value('lightgray'))
        )
        .transform_filter(alt.datum.gate_status == "Closed")
    )

    # modeled_historic_below_wl = alt.Chart(filtered_df).mark_text(align='right').encode(
    #         y=alt.Y('wl_stat:N', axis=None),
    #             text=alt.Text('below_wl:N')
    #         ).transform_filter(
    #             interval
    #         ).transform_filter(
    #             "datum.Modeled_Historic < 2.3"
    #         ).transform_aggregate(
    #             total_time_below_wl='sum(time_unit)'
    #         ).transform_calculate(
    #             below_wl='format(datum.total_time_below_wl, ".2f") + " hour"'
    #         ).properties(
    #             title=alt.Title(text='Modeled Historic Below Water Level Compliance', align='center')
    #         )

    avg_stage = avg_stage.properties(width=200, height=100)
    scenario_duration_below_wl = scenario_duration_below_wl.properties(
        width=200, height=100
    )
    # modeled_historic_below_wl = modeled_historic_below_wl.properties(width=200, height=100)
    # (base, points, yrule, rules, area_dgl_true
    min_stage_chart = alt.layer(
        base_elevation,
        points_elev,
        yrule_wl,
        yrule_wl_text,
        rules_elev,
        area_gate_true,
        text,
    ).properties(width=700, height=400, title=f"Modeled Water Level at {gate}")
    return min_stage_chart

