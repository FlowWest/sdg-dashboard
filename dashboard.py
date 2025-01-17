import streamlit as st
import pandas as pd
import altair as alt
import pickle
from figures_functions import *
#import plotly.express as px
import datetime
import numpy as np
from streamlit_folium import st_folium, folium_static

#TODO: add border to top barchart
#TODO: add elevation graph based on week

# Title and description
st.set_page_config(layout="wide")

# pg = st.navigation([
#     st.Page("page1.py", title="First page", icon="🔥"),
#     # st.Page(page2, title="Second page", icon=":material/favorite:"),
# ])
# pg.run()

st.title("Exploratory Data Visualizations for SDG Analysis")
st.write("Upload your data and explore interactive visualizations.")
# File uploader
# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#look for multi files
uploaded_file = st.file_uploader("Upload Pickle file", type="pkl")

if uploaded_file:
    #--------------------------------------------------------------------------------------------------------------------------------
    #Data wrangling
    multi_model_data = pickle.load(uploaded_file)
    
#-----------------------------------------------------------------------------------------------------------------------------------
#     df = full_merged_df.rename(
#     columns={
#         "GLC": "Flow (ft/s)",
#         "datetime": "Datetime",
#         "Velocity_Category": "Velocity Category",
#         "consecutive_groups": "Consecutive Groups",
#         "min_datetime": "Flow Duration Min Datetime",
#         "max_datetime": "Flow Duration Max Datetime",
#         "streak_duration": "Streak Duration (hrs)",
#         "gate_min_datetime": "Gate Min Datetime",
#         "gate_max_datetime": "Gate Max Datetime",
#         "gate_count": "Gate Count",
#         "gate_streak_duration": "Gate Streak Duration (hrs)",
#         "time_unit": "Time Unit (hrs)",
#     }
#     )
#     df = df.reset_index(drop=True)
    
    #-------------------------------------------------------------------------------------------------------------------------------
    # Markdown
    # location_gate = {
    #     "GLC":"Grantline",
    #     "MID":"MiddleRiver",
    #     "OLD":"OldRiver",
    # }
    models = multi_model_data.keys()
    with st.sidebar:
        selected_model = st.selectbox('Select Model:', models)
        years = multi_model_data[selected_model]["GLC"]["gate_operation_data"]["datetime"].dt.year.unique().tolist()
        years.append("None")
        selected_option = st.selectbox('Select Year:', years)
        if selected_option != "None":
            selected_year = int(selected_option)
        else:
            selected_year = None
    # st.sidebar.write("### Filter by Date Range")
    # start_date, end_date = st.sidebar.date_input(
    #     "Select a date range:",
    #     [multi_model_data[selected_model]["GLC"]["gate_operation_data"]["datetime"].min().date(), 
    #     multi_model_data[selected_model]["GLC"]["gate_operation_data"]["datetime"].max().date()],
    #     min_value=multi_model_data[selected_model]["GLC"]["gate_operation_data"]["datetime"].min().date(),
    #     max_value=multi_model_data[selected_model]["GLC"]["gate_operation_data"]["datetime"].max().date()
    # )

    glc_full_merged_df = post_process_full_data(multi_model_data, 
                                                selected_model, 
                                                "GLC", 
                                                year=selected_year)
    glc_full_merged_df = glc_full_merged_df.rename(columns={"value": "velocity"})
    glc_hydro_df = post_process_hydro_data(multi_model_data, selected_model, "GLC", selected_year)
    glc_avg_daily_velocity = calc_avg_daily_vel(glc_full_merged_df)
    glc_avg_daily_gate = calc_avg_daily_gate(glc_full_merged_df)
    glc_total_daily_velocity = calc_avg_len_consec_vel(glc_full_merged_df)
    glc_total_daily_gate = calc_avg_len_consec_gate(glc_full_merged_df)

    old_full_merged_df = post_process_full_data(multi_model_data, selected_model, "OLD", year=selected_year)
    old_full_merged_df = old_full_merged_df.rename(columns={"value": "velocity"})
    old_hydro_df = post_process_hydro_data(multi_model_data, selected_model, "OLD", selected_year)
    old_avg_daily_velocity = calc_avg_daily_vel(old_full_merged_df)
    old_avg_daily_gate = calc_avg_daily_gate(old_full_merged_df)
    old_total_daily_velocity = calc_avg_len_consec_vel(old_full_merged_df)
    old_total_daily_gate = calc_avg_len_consec_gate(old_full_merged_df)

    mid_full_merged_df = post_process_full_data(multi_model_data, selected_model, "MID", year=selected_year)
    mid_hydro_df = post_process_hydro_data(multi_model_data, selected_model, "MID", selected_year)
    mid_full_merged_df = mid_full_merged_df.rename(columns={"value": "velocity"})
    mid_avg_daily_velocity = calc_avg_daily_vel(mid_full_merged_df)
    mid_avg_daily_gate = calc_avg_daily_gate(mid_full_merged_df)
    mid_total_daily_velocity = calc_avg_len_consec_vel(mid_full_merged_df)
    mid_total_daily_gate = calc_avg_len_consec_gate(mid_full_merged_df)

    glc_min_date = min(glc_full_merged_df['date'])
    glc_max_date = max(glc_full_merged_df['date']) 

    glc_min_velocity = min(glc_full_merged_df['velocity'])
    glc_max_velocity = max(glc_full_merged_df['velocity'])

    velocity_summary_stats_title = f"Summary stats of fish passage from {glc_min_date} to {glc_max_date}."
    gate_summary_stats_title = f"Summary stats of upstream of gate from {glc_min_date} to {glc_max_date}."
    min_max_summary_title = f"Min max stats of fish passage from {glc_min_date} to {glc_max_date}."
    
    velocity_summary_data = {
        "Location": [
            location_gate[glc_full_merged_df['gate'][0]],
            location_gate[mid_full_merged_df['gate'][0]],
            location_gate[old_full_merged_df['gate'][0]]
        ],
        f"Average Daily Time (Hours) {glc_avg_daily_velocity['Velocity_Category'][0]}":[
            round(glc_avg_daily_velocity['time_unit'][0], 2),
            round(mid_avg_daily_velocity['time_unit'][0], 2),
            round(old_avg_daily_velocity['time_unit'][0], 2),
        ],
        f"Average Daily Time (Hours) {glc_avg_daily_velocity['Velocity_Category'][1]}":[
            round(glc_avg_daily_velocity['time_unit'][1], 2),
            round(mid_avg_daily_velocity['time_unit'][1], 2),
            round(old_avg_daily_velocity['time_unit'][1], 2),
        ],
        f"Average Streak Duration (Hours) {glc_total_daily_velocity['Velocity_Category'][0]}":[
            round(glc_total_daily_velocity['daily_average_time_per_consecutive_group'][0], 2),
            round(mid_total_daily_velocity['daily_average_time_per_consecutive_group'][0], 2),
            round(old_total_daily_velocity['daily_average_time_per_consecutive_group'][0], 2),
        ],
        f"Average Streak Duration (Hours) {glc_total_daily_velocity['Velocity_Category'][1]}":[
            round(glc_total_daily_velocity['daily_average_time_per_consecutive_group'][1], 2),
            round(mid_total_daily_velocity['daily_average_time_per_consecutive_group'][1], 2),
            round(old_total_daily_velocity['daily_average_time_per_consecutive_group'][1], 2),
        ]
    }

    gate_summary_data = {
        "Location": [
            glc_full_merged_df['gate'][0],
            mid_full_merged_df['gate'][0],
            old_full_merged_df['gate'][0]
        ],
        f"Average Daily {glc_avg_daily_gate['gate_status'][0]} Time (Hours) for gate":[
            round(glc_avg_daily_gate['time_unit'][0], 2),
            round(mid_avg_daily_gate['time_unit'][0], 2),
            round(old_avg_daily_gate['time_unit'][0], 2),
        ],
        f"Average Daily {glc_avg_daily_gate['gate_status'][1]} Time (Hours) for gate":[
            round(glc_avg_daily_gate['time_unit'][1], 2),
            round(mid_avg_daily_gate['time_unit'][1], 2),
            round(old_avg_daily_gate['time_unit'][1], 2),
        ],
        f"Average {glc_total_daily_gate['gate_status'][0]} Duration (Hours) Per Streak":[
            round(glc_total_daily_gate['daily_average_time_per_consecutive_gate'][0], 2),
            round(mid_total_daily_gate['daily_average_time_per_consecutive_gate'][0], 2),
            round(old_total_daily_gate['daily_average_time_per_consecutive_gate'][0], 2),
        ],
        f"Average {glc_total_daily_gate['gate_status'][1]} Duration (Hours) Per Streak":[
            round(glc_total_daily_gate['daily_average_time_per_consecutive_gate'][1], 2),
            round(mid_total_daily_gate['daily_average_time_per_consecutive_gate'][1], 2),
            round(old_total_daily_gate['daily_average_time_per_consecutive_gate'][1], 2),
        ]
    }
    
    min_max_summary = {
        "Location": [
            location_gate[glc_full_merged_df['gate'][0]],
            location_gate[mid_full_merged_df['gate'][0]],
            location_gate[old_full_merged_df['gate'][0]]
        ],
        "Minimum velocity through fish passage (ft/s)":[
            round(min(glc_full_merged_df['velocity']), 2),
            round(min(mid_full_merged_df['velocity']), 2),
            round(min(old_full_merged_df['velocity']), 2),
        ],
        "Maximum velocity through fish passage (ft/s)":[
            round(max(glc_full_merged_df['velocity']), 2),
            round(max(mid_full_merged_df['velocity']), 2),
            round(max(old_full_merged_df['velocity']), 2),
        ]
    }

       # Create a DataFrame
    velocity_summary_df = pd.DataFrame(velocity_summary_data)
    gate_summary_df = pd.DataFrame(gate_summary_data)
    min_max_vel_summary_df = pd.DataFrame(min_max_summary)


    
    glc_chart = generate_velocity_gate_charts(glc_full_merged_df)
    mid_chart = generate_velocity_gate_charts(mid_full_merged_df)
    old_chart = generate_velocity_gate_charts(old_full_merged_df, legend=True)
    st.write("### Data Preview")
    data_preview_glc, data_preview_mid, data_preview_old = st.tabs(["GLC", "MID", "OLD"])
    data_preview_glc.dataframe(glc_full_merged_df.head(20).style.format(precision=2).set_table_styles(
        [{
            'selector': 'thead th',
            'props': [('background-color', '#4CAF50'), ('color', 'white'), ('text-align', 'center')]
        },
         {
            'selector': 'tbody tr:hover',
            'props': [('background-color', '#f5f5f5')]
        }]
    ), use_container_width=True)

    data_preview_mid.dataframe(mid_full_merged_df.head(20).style.format(precision=2).set_table_styles(
        [{
            'selector': 'thead th',
            'props': [('background-color', '#4CAF50'), ('color', 'white'), ('text-align', 'center')]
        },
         {
            'selector': 'tbody tr:hover',
            'props': [('background-color', '#f5f5f5')]
        }]
    ), use_container_width=True)

    data_preview_old.dataframe(old_full_merged_df.head(20).style.format(precision=2).set_table_styles(
        [{
            'selector': 'thead th',
            'props': [('background-color', '#4CAF50'), ('color', 'white'), ('text-align', 'center')]
        },
         {
            'selector': 'tbody tr:hover',
            'props': [('background-color', '#f5f5f5')]
        }]
    ), use_container_width=True)

    # st.write("### Gate and Channel Locations") 

    # shapefile_paths = [
    #     "data-raw/MSS_nodes/dsm2_nodes_newcs_extranodes.shp",
    #     "data-raw/fc2024.01_chan/FC2024.01_channels_centerlines.shp"
    # ]
    # nodes = gpd.read_file(shapefile_paths[0])
    # channels = gpd.read_file(shapefile_paths[1])
    # nodes_to_highlight = [112, 176, 69]
    # nodes_filter = nodes[nodes['id'].isin(nodes_to_highlight)]
    # channels_with_numbers = pd.read_csv('data-raw/channel_names_from_h5.csv')
    # channels_with_numbers = channels_with_numbers.rename(columns={'chan_no': 'id'})

    # channels_merge = pd.merge(
    #     channels,
    #     channels_with_numbers,
    #     how='left',
    #     left_on='id',
    #     right_on='id'
    # )

    # filtered_channels = channels_merge[channels_merge['id'].isin([211, 79, 134])]

    # Generate the map
    # gdfs, all_centroids= process_shapefiles(shapefile_paths)
    # nodes_filter, filtered_channels = transform_and_filter_geometries(nodes_filter, filtered_channels)
    # avg_lat, avg_lon = calculate_avg_lat_long(all_centroids)
    # map_object = create_multi_layer_map(gdfs=gdfs, avg_lat=avg_lat, avg_lon=avg_lon, filtered_gdf=nodes_filter, filtered_polylines=filtered_channels)
    # col1, col2, col3 = st.columns([1, 3, 1])
    # with col2:
    #     st_map = st_folium(map_object, width=1200, height=500)

# Display the selected data (If any)
    # st.dataframe(st.session_state.map_data)
    st.write("### Daily Gate Status Duration vs Daily Velocity Flow Duration")
    
    viz_1_tab1, viz_1_tab2 = st.tabs(["🗃 Data Summary", "📈 Chart"])
    # viz_1_tab1.write("### Data Summary")
    viz_1_tab1.write(f"##### {velocity_summary_stats_title}")
    viz_1_tab1.dataframe(velocity_summary_df.style.highlight_max(
        subset=velocity_summary_df.columns[1:],
        color = "#ffffc5").format(precision=2))   
    viz_1_tab1.write("")
    viz_1_tab1.write(f"##### {min_max_summary_title}")
    viz_1_tab1.dataframe(min_max_vel_summary_df.style.highlight_max(
        subset=min_max_vel_summary_df.columns[1:],
        color = "#ffffc5").format(precision=2))
    viz_1_tab1.write("")
    viz_1_tab1.write(f"##### {gate_summary_stats_title}")
    viz_1_tab1.dataframe(gate_summary_df.style.highlight_max(subset=gate_summary_df.columns[1:],
                                                             color = "#ffffc5").format(precision=2))
    # Altair Visualization
    # st.write('#')    
    # st.altair_chart(combined_chart, use_container_width=True, theme=None)
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
    st.write('###')
    st.write("### Flow Velocity and Gate Status Zoomed")
    # drop_down_week = glc_full_merged_df['week'].unique().tolist()
    # week_to_date_mapping = glc_full_merged_df.groupby("week")["date"].min()
    # week_to_date_dict = week_to_date_mapping.to_dict()
    # drop_down_options = [
    #     f"Week {week} (Start Date: {date})"
    #     for week, date in week_to_date_mapping.items()
    # ]
    # selected_option = st.selectbox('Select Week:', drop_down_options)
    # selected_week = int(selected_option.split()[1])
    default_start_date = pd.to_datetime(glc_full_merged_df['date'].min())
    default_end_date = pd.to_datetime(glc_full_merged_df['date'].min()) + datetime.timedelta(days=7)
    
    start_date = default_start_date
    end_date = default_end_date

    if "selected_date_range" not in st.session_state:
        st.session_state.selected_date_range = None

    def submit_date_range():
        st.session_state.selected_date_range = st.session_state.date_range_input
    

    st.date_input("Pick a date range:", value=(default_start_date, default_end_date), key="date_range_input", help="Select start and end dates")
    st.button("Filter Data", on_click=submit_date_range)
    if st.session_state.selected_date_range:
        start_date, end_date = st.session_state.selected_date_range
        if start_date and end_date:
            st.write(f"Date Filtered.")

    start_date = str(start_date)
    end_date = str(end_date)

    filtered_glc_df = glc_full_merged_df[(glc_full_merged_df['date'] >= start_date) & (glc_full_merged_df['date'] <= end_date)]
    filtered_mid_df = mid_full_merged_df[(mid_full_merged_df['date'] >= start_date) & (mid_full_merged_df['date'] <= end_date)]
    filtered_old_df = old_full_merged_df[(old_full_merged_df['date'] >= start_date) & (old_full_merged_df['date'] <= end_date)]

    filtered_glc_hydro_df = glc_hydro_df[(glc_hydro_df['datetime'] >= start_date) & (glc_hydro_df['datetime'] <= end_date)]
    filtered_mid_hydro_df = mid_hydro_df[(mid_hydro_df['datetime'] >= start_date) & (mid_hydro_df['datetime'] <= end_date)]
    filtered_old_hydro_df = old_hydro_df[(old_hydro_df['datetime'] >= start_date) & (old_hydro_df['datetime'] <= end_date)]
# #-------------------------------------------------------------------------------------------------------
    summary_stats_title = f"Summary stats from {start_date} to {end_date}."
    filtered_glc_avg_daily_velocity = calc_avg_daily_vel(filtered_glc_df)
    filtered_glc_avg_daily_gate = calc_avg_daily_gate(filtered_glc_df)
    
    filtered_old_avg_daily_velocity = calc_avg_daily_vel(filtered_old_df)
    filtered_old_avg_daily_gate = calc_avg_daily_gate(filtered_old_df)
    
    filtered_mid_avg_daily_velocity = calc_avg_daily_vel(filtered_mid_df)
    filtered_mid_avg_daily_gate = calc_avg_daily_gate(filtered_mid_df)
    
    # weekly_daily_velocity = filtered_df.groupby(["date", "Velocity_Category"])["time_unit"].sum().reset_index()
    # weekly_avg_daily_velocity = weekly_daily_velocity.groupby("Velocity_Category")['time_unit'].mean().reset_index()
    # weekly_daily_gate = filtered_df.groupby(["date","gate_status"])["time_unit"].sum().reset_index()
    # weekly_avg_daily_gate = weekly_daily_gate.groupby("gate_status")['time_unit'].mean().reset_index()
    weekly_min_date = min(filtered_glc_df['date'])
    weekly_max_date = max(filtered_glc_df['date'])

    weekly_summary_data = {
        "Location": [
            location_gate[glc_full_merged_df['gate'][0]],
            location_gate[mid_full_merged_df['gate'][0]],
            location_gate[old_full_merged_df['gate'][0]]
        ],
        f"Average Daily Time (Hours) {filtered_glc_avg_daily_velocity['Velocity_Category'][0]}":[
            round(filtered_glc_avg_daily_velocity['time_unit'][0], 2),
            round(filtered_mid_avg_daily_velocity['time_unit'][0], 2),
            round(filtered_old_avg_daily_velocity['time_unit'][0], 2),
        ],
        f"Average Daily Time (Hours) {filtered_glc_avg_daily_velocity['Velocity_Category'][1]}":[
            round(filtered_glc_avg_daily_velocity['time_unit'][1], 2),
            round(filtered_mid_avg_daily_velocity['time_unit'][1], 2),
            round(filtered_old_avg_daily_velocity['time_unit'][1], 2),
        ],
        f"Average Daily {filtered_glc_avg_daily_gate['gate_status'][0]} Time (Hours) for gate":[
            round(filtered_glc_avg_daily_gate['time_unit'][0], 2),
            round(filtered_mid_avg_daily_gate['time_unit'][0], 2),
            round(filtered_old_avg_daily_gate['time_unit'][0], 2),
        ],
        f"Average Daily {filtered_glc_avg_daily_gate['gate_status'][1]} Time (Hours) for gate":[
            round(filtered_glc_avg_daily_gate['time_unit'][1], 2),
            round(filtered_mid_avg_daily_gate['time_unit'][1], 2),
            round(filtered_old_avg_daily_gate['time_unit'][1], 2),
        ],

    }
    # def color_coding(row):
    #     if row['Average Daily Time (Hours) Over 8ft/s'] > 15:
    #         return ['background-color: red'] * len(row)
    #     else:
    #         return ['background-color: green'] * len(row)
# Create a DataFrame
    viz_2_tab1, viz_2_tab2 = st.tabs(["🗃 Data Summary", "📈 Chart"])
    # try:
    weekly_summary_df = pd.DataFrame(weekly_summary_data)
    # styled_df = weekly_summary_df.style.apply(color_coding, axis=1)
    # styled_html = styled_df.to_html()

    # weekly_summary_df.iloc[:, 1:4] = weekly_summary_df[1:].apply(pd.to_numeric)
    viz_2_tab1.write(f"##### {summary_stats_title}")
    # .set_properties(**{'font-weight': 'bold'}, subset=df.columns
    # st.markdown(
    #     styled_html,
    #     unsafe_allow_html=True
    # )
    viz_2_tab1.dataframe(weekly_summary_df.style.highlight_max(subset=weekly_summary_df.columns[1:],
                                                               color = "#ffffc5").format(precision=2))
    # except:
        # "Missing data to generate summary data for this time period. "
    # #-------------------------------------------------------------------------------------------------------
    # # Create an Altair chart using the filtered data
    # # Define a colorblind-friendly palette
    # Display the chart in Streamlit
    
    # st.altair_chart(daily_velocity, use_container_width=False)
    glc_zoomed_vel_chart  = generate_zoomed_velocity_charts(filtered_glc_df)
    mid_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_mid_df)
    old_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_old_df)
    # st.write("#")
    col1, col2, col3 = viz_2_tab2.columns([10, 10, 10], gap="small")
    with col1:
        st.altair_chart(glc_zoomed_vel_chart, 
                        # use_container_width=True, 
                        theme=None)
        # st.altair_chart(glc_zoomed_vel_chart[1], use_container_width=True, theme=None)
    with col2:
        st.altair_chart(mid_zoomed_vel_chart, 
                        # use_container_width=True, 
                        theme=None)
    with col3:
        st.altair_chart(old_zoomed_vel_chart, 
                        # use_container_width=True, 
                        theme=None)
    # col1, col2, col3 = st.columns([10, 10, 10], gap="small")
    # with col1:
    #     st.altair_chart(glc_zoomed_vel_chart[1], use_container_width=True, theme=None)
    #     # st.altair_chart(glc_zoomed_vel_chart[1], use_container_width=True, theme=None)
    # with col2:
    #     st.altair_chart(mid_zoomed_vel_chart[1], use_container_width=True, theme=None)
    # with col3:
    #     st.altair_chart(old_zoomed_vel_chart[1], use_container_width=True, theme=None)

    glc_zoomed_hydro_chart  = generate_water_level_chart(filtered_glc_hydro_df,filtered_glc_df)
    mid_zoomed_hydro_chart = generate_water_level_chart(filtered_mid_hydro_df, filtered_mid_df)
    old_zoomed_hydro_chart = generate_water_level_chart(filtered_old_hydro_df, filtered_old_df)
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
    # st.altair_chart(combined_chart, use_container_width=False, theme=None)
    # st.altair_chart(combined_elev_chart, use_container_width=False, theme=None)
    # st.altair_chart(joint_chart, use_container_width=False, theme=None)

 

else:
    st.write("Please upload a Pickle file to see the visualization.")
