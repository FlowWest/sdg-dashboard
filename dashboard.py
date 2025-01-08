import streamlit as st
import pandas as pd
import altair as alt
import pickle
from figures_functions import *
import plotly.express as px
import datetime

#TODO: add border to top barchart
#TODO: add elevation graph based on week

# Title and description
st.set_page_config(layout="wide")
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
            f"{glc_avg_daily_velocity['time_unit'][0]:.2f}",
            f"{mid_avg_daily_velocity['time_unit'][0]:.2f}",
            f"{old_avg_daily_velocity['time_unit'][0]:.2f}",
        ],
        f"Average Daily Time (Hours) {glc_avg_daily_velocity['Velocity_Category'][1]}":[
            f"{glc_avg_daily_velocity['time_unit'][1]:.2f}",
            f"{mid_avg_daily_velocity['time_unit'][1]:.2f}",
            f"{old_avg_daily_velocity['time_unit'][1]:.2f}",
        ],
        f"Average Streak Duration (Hours) {glc_total_daily_velocity['Velocity_Category'][0]}":[
            f"{glc_total_daily_velocity['daily_average_time_per_consecutive_group'][0]:.2f}",
            f"{mid_total_daily_velocity['daily_average_time_per_consecutive_group'][0]:.2f}",
            f"{old_total_daily_velocity['daily_average_time_per_consecutive_group'][0]:.2f}",
        ],
        f"Average Streak Duration (Hours) {glc_total_daily_velocity['Velocity_Category'][1]}":[
            f"{glc_total_daily_velocity['daily_average_time_per_consecutive_group'][1]:.2f}",
            f"{mid_total_daily_velocity['daily_average_time_per_consecutive_group'][1]:.2f}",
            f"{old_total_daily_velocity['daily_average_time_per_consecutive_group'][1]:.2f}",
        ]
    }

    gate_summary_data = {
        "Location": [
            glc_full_merged_df['gate'][0],
            mid_full_merged_df['gate'][0],
            old_full_merged_df['gate'][0]
        ],
        f"Average Daily {glc_avg_daily_gate['gate_status'][0]} Time (Hours) for gate":[
            f"{glc_avg_daily_gate['time_unit'][0]:.2f}",
            f"{mid_avg_daily_gate['time_unit'][0]:.2f}",
            f"{old_avg_daily_gate['time_unit'][0]:.2f}",
        ],
        f"Average Daily {glc_avg_daily_gate['gate_status'][1]} Time (Hours) for gate":[
            f"{glc_avg_daily_gate['time_unit'][1]:.2f}",
            f"{mid_avg_daily_gate['time_unit'][1]:.2f}",
            f"{old_avg_daily_gate['time_unit'][1]:.2f}",
        ],
        f"Average {glc_total_daily_gate['gate_status'][0]} Duration (Hours) Per Streak":[
            f"{glc_total_daily_gate['daily_average_time_per_consecutive_gate'][0]:.2f}",
            f"{mid_total_daily_gate['daily_average_time_per_consecutive_gate'][0]:.2f}",
            f"{old_total_daily_gate['daily_average_time_per_consecutive_gate'][0]:.2f}",
        ],
        f"Average {glc_total_daily_gate['gate_status'][1]} Duration (Hours) Per Streak":[
            f"{glc_total_daily_gate['daily_average_time_per_consecutive_gate'][1]:.2f}",
            f"{mid_total_daily_gate['daily_average_time_per_consecutive_gate'][1]:.2f}",
            f"{old_total_daily_gate['daily_average_time_per_consecutive_gate'][1]:.2f}",
        ]
    }
    
    min_max_summary = {
        "Location": [
            location_gate[glc_full_merged_df['gate'][0]],
            location_gate[mid_full_merged_df['gate'][0]],
            location_gate[old_full_merged_df['gate'][0]]
        ],
        "Minimum velocity through fish passage":[
            f"{min(glc_full_merged_df['velocity']):.2f} ft/s",
            f"{min(mid_full_merged_df['velocity']):.2f} ft/s",
            f"{min(old_full_merged_df['velocity']):.2f} ft/s",
        ],
        "Maximum velocity through fish passage":[
            f"{max(glc_full_merged_df['velocity']):.2f} ft/s",
            f"{max(mid_full_merged_df['velocity']):.2f} ft/s",
            f"{max(old_full_merged_df['velocity']):.2f} ft/s",
        ]
    }

       # Create a DataFrame
    velocity_summary_df = pd.DataFrame(velocity_summary_data)
    gate_summary_df = pd.DataFrame(gate_summary_data)
    min_max_vel_summary_df = pd.DataFrame(min_max_summary)
    
    glc_chart = generate_velocity_gate_charts(glc_full_merged_df)
    old_chart = generate_velocity_gate_charts(old_full_merged_df)
    mid_chart = generate_velocity_gate_charts(mid_full_merged_df)
    st.write("### Data Preview")
    st.dataframe(glc_full_merged_df.head(20).style.format(precision=2).set_table_styles(
        [{
            'selector': 'thead th',
            'props': [('background-color', '#4CAF50'), ('color', 'white'), ('text-align', 'center')]
        },
         {
            'selector': 'tbody tr:hover',
            'props': [('background-color', '#f5f5f5')]
        }]
    ), use_container_width=True)

    # st.write("### Interactive Map - Click a Point") 
    # data = pd.DataFrame({
    #     "gates": ["GLC", "OLD", "MID"],
    #     "latitude": [34.07105502661072, 34.055021339285005, 34.07412241104673],
    #     "longitude": [-118.33807459509656, -118.25021296787665, -118.25840626969342],
    #     # "value": [100, 200, 300]
    # })

    # # Create an interactive Plotly map
    # fig_map = px.scatter_mapbox(
    #     data,
    #     lat="latitude",
    #     lon="longitude",
    #     size=[50, 50, 50],
    #     color="gates",
    #     size_max=100,
    #     zoom=12,
    #     mapbox_style="carto-positron"  # Map style
    # )

    # event = st.plotly_chart(
    #     fig_map,
    #     on_select="rerun",
    #     selection_mode=["box", "points"],
    #     key="map_data",  # Store selection in session_state
    # )
    # event
    # st.write(st.session_state.map_data)



# Display the selected data (If any)
    # st.dataframe(st.session_state.map_data)
    
    st.write("### Data Summary")
    st.write(velocity_summary_stats_title)
    st.table(velocity_summary_df)   
    st.write("")
    st.write(min_max_summary_title)
    st.table(min_max_vel_summary_df)
    st.write("")
    st.write(gate_summary_stats_title)
    st.table(gate_summary_df)
    # Altair Visualization
    st.write('#')    
    st.write("### Visualization 1: Daily Gate Status Duration vs Daily Velocity Flow Duration")
    # st.altair_chart(combined_chart, use_container_width=True, theme=None)
    col1, col2, col3 = st.columns([3, 3, 3], gap="small")
    with col1:
        st.altair_chart(glc_chart, use_container_width=True, theme=None)
    with col2:
        st.altair_chart(mid_chart, use_container_width=True, theme=None)
    with col3:
        st.altair_chart(old_chart, use_container_width=True, theme=None)
    st.write('###')
    st.write("### Visualization 2: Flow Velocity and Gate Status Zoomed")
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
            f"{filtered_glc_avg_daily_velocity['time_unit'][0]:.2f}",
            f"{filtered_mid_avg_daily_velocity['time_unit'][0]:.2f}",
            f"{filtered_old_avg_daily_velocity['time_unit'][0]:.2f}",
        ],
        f"Average Daily Time (Hours) {filtered_glc_avg_daily_velocity['Velocity_Category'][1]}":[
            f"{filtered_glc_avg_daily_velocity['time_unit'][1]:.2f}",
            f"{filtered_mid_avg_daily_velocity['time_unit'][1]:.2f}",
            f"{filtered_old_avg_daily_velocity['time_unit'][1]:.2f}",
        ],
        f"Average Daily {filtered_glc_avg_daily_gate['gate_status'][0]} Time (Hours) for gate":[
            f"{filtered_glc_avg_daily_gate['time_unit'][0]:.2f}",
            f"{filtered_mid_avg_daily_gate['time_unit'][0]:.2f}",
            f"{filtered_old_avg_daily_gate['time_unit'][0]:.2f}",
        ],
        f"Average Daily {filtered_glc_avg_daily_gate['gate_status'][1]} Time (Hours) for gate":[
            f"{filtered_glc_avg_daily_gate['time_unit'][1]:.2f}",
            f"{filtered_mid_avg_daily_gate['time_unit'][1]:.2f}",
            f"{filtered_old_avg_daily_gate['time_unit'][1]:.2f}",
        ],

    }

# Create a DataFrame
    weekly_summary_df = pd.DataFrame(weekly_summary_data)
    st.write(summary_stats_title)
    st.table(weekly_summary_df)
    # #-------------------------------------------------------------------------------------------------------
    # # Create an Altair chart using the filtered data
    # # Define a colorblind-friendly palette

    # elev_cols = ["FP2VMa", "Modeled_Historic"]

    # base_elevation = alt.Chart(filtered_glc_df).mark_line().encode(
    #     x=alt.X('yearmonthdatehoursminutes(datetime):T', title='Date', axis=alt.Axis(format='%b %d, %Y', labelAngle=-45)),
    #     y=alt.Y('value:Q', title='Feet'),
    #     color='model:N'
    # ).transform_fold(
    #     ['FP2VMa', 'Modeled_Historic'],  # Columns to be "melted" into a long format
    #     as_=['model', 'value']  # New column names
    # ).add_params(
    #     interval
    # ).properties(
    #     title="Weekly Summary of Stage, upstream of gates @ DGL"
    # )
    # yrule_wl = alt.Chart(filtered_glc_df).mark_rule(color = "purple", strokeDash=[12, 6], size=1.5).encode(
    #         y=alt.datum(2.3)
    # )
    # yrule_wl_text = alt.Chart(filtered_glc_df).mark_text(
    #     text="Water Level Compliance",
    #     align="left",
    #     baseline="bottom",
    #     fontSize=12,
    #     color="grey",
    #     dx=5  # Offset text slightly to the right of the rule
    # ).encode(
    #     y=alt.datum(2.3)  # Same y position as the rule
    # )
    # nearest_elev = alt.selection_point(nearest=True, on="pointerover",
    #                               fields=["datetime"], empty=False)
    # points_elev = base_elevation.mark_point().encode(
    #     opacity=alt.condition(nearest_elev, alt.value(1), alt.value(0))
    # )
    # rules_elev = alt.Chart(filtered_glc_df).mark_rule(color="gray", opacity=0).encode(
    #     x="datetime:T",
    #     opacity=alt.condition(nearest_elev, alt.value(0.3), alt.value(0)),
    # ).add_params(nearest_elev)
    # when_near = alt.when(nearest_elev)
    # text = base_elevation.mark_text(
    #     align="left", dx=5, dy=-5
    # ).transform_calculate(
    #     label='format(datum.value, ".2f") + " feet"'
    # ).encode(
    #     text=when_near.then("label:N").otherwise(alt.value(" "))
    # )
    # average_scenario_stage = alt.Chart(filtered_glc_df).mark_text(align='right').encode(
    #         y=alt.Y('stat:N', axis=None),
    #         text=alt.Text('summary:N')
    #     ).transform_filter(
    #         interval
    #     ).transform_aggregate(
    #         average_stage='mean(FP2VMa)'
    #     ).transform_fold(
    #         ['average_stage'],  # Separate each statistic
    #         as_=['stat', 'value']
    #     ).transform_calculate(
    #         summary='format(datum.average_stage, ".2f") + " feet"'
    #     )
    # avg_stage = average_scenario_stage.encode(text='summary:N').properties(
    #         title=alt.Title(text='Average FP2VMa Minimum Stage', align='center')
    # )
    # scenario_duration_below_wl = alt.Chart(filtered_glc_df).mark_text(align='right').encode(
    #     y=alt.Y('wl_stat:N', axis=None),
    #         text=alt.Text('below_wl:N')
    #     # ).transform_filter(
    #     #     interval
    #     ).transform_filter(
    #         "datum.FP2VMa < 2.3"
    #     ).transform_aggregate(
    #         total_time_below_wl='sum(time_unit)'
    #     ).transform_calculate(
    #         below_wl='format(datum.total_time_below_wl, ".2f") + " hour"'
    #     ).properties(
    #         title=alt.Title(text='Scenario Below Water Level Compliance', align='center')
    #     )

    # modeled_historic_below_wl = alt.Chart(filtered_glc_df).mark_text(align='right').encode(
    #     y=alt.Y('wl_stat:N', axis=None),
    #         text=alt.Text('below_wl:N')
    #     ).transform_filter(
    #         interval
    #     ).transform_filter(
    #         "datum.Modeled_Historic < 2.3"
    #     ).transform_aggregate(
    #         total_time_below_wl='sum(time_unit)'
    #     ).transform_calculate(
    #         below_wl='format(datum.total_time_below_wl, ".2f") + " hour"'
    #     ).properties(
    #         title=alt.Title(text='Modeled Historic Below Water Level Compliance', align='center')
    #     )
    
    # avg_stage = avg_stage.properties(width=200, height=100)
    # scenario_duration_below_wl = scenario_duration_below_wl.properties(width=200, height=100)
    # modeled_historic_below_wl = modeled_historic_below_wl.properties(width=200, height=100)
    # # (base, points, yrule, rules, area_gate_status_true
    # weekly_min_stage_chart = alt.layer(base_elevation, 
    #                           points_elev, 
    #                           yrule_wl, 
    #                           yrule_wl_text,
    #                           rules_elev,
    #                           area_dgl_true,
    #                           text
    # ).properties(width=650, height=400)
    
    # combined_elev_text = alt.vconcat(
    #     avg_stage,
    #     scenario_duration_below_wl,
    #     modeled_historic_below_wl
    # )
    # combined_elev_chart = alt.hconcat(
    #     weekly_min_stage_chart,
    #     combined_elev_text
    # )
    
    # joint_chart = alt.vconcat(
    #     combined_elev_chart,
    #     combined_chart
    # )
    # Display the chart in Streamlit
    
    # st.altair_chart(daily_velocity, use_container_width=False)
    glc_zoomed_vel_chart  = generate_zoomed_velocity_charts(filtered_glc_df)
    mid_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_mid_df)
    old_zoomed_vel_chart = generate_zoomed_velocity_charts(filtered_old_df)
    st.write("#")
    col1, col2, col3 = st.columns([3, 3, 3], gap="small")
    with col1:
        st.altair_chart(glc_zoomed_vel_chart, use_container_width=False, theme=None)
    with col2:
        st.altair_chart(mid_zoomed_vel_chart, use_container_width=False, theme=None)
    with col3:
        st.altair_chart(old_zoomed_vel_chart, use_container_width=False, theme=None)

    glc_zoomed_hydro_chart  = generate_water_level_chart(filtered_glc_hydro_df,filtered_glc_df)
    mid_zoomed_hydro_chart = generate_water_level_chart(filtered_mid_hydro_df, filtered_mid_df)
    old_zoomed_hydro_chart = generate_water_level_chart(filtered_old_hydro_df, filtered_old_df)
    st.write("#")
    col1, col2, col3 = st.columns([3, 3, 3], gap="small")
    with col1:
        st.altair_chart(glc_zoomed_hydro_chart, use_container_width=False, theme=None)
    with col2:
        st.altair_chart(mid_zoomed_hydro_chart, use_container_width=False, theme=None)
    with col3:
        st.altair_chart(old_zoomed_hydro_chart, use_container_width=False, theme=None)
    # st.altair_chart(combined_chart, use_container_width=False, theme=None)
    # st.altair_chart(combined_elev_chart, use_container_width=False, theme=None)
    # st.altair_chart(joint_chart, use_container_width=False, theme=None)

 

else:
    st.write("Please upload a Pickle file to see the visualization.")
