import streamlit as st
import pandas as pd
import altair as alt

# Title and description
# st.set_page_config(layout="wide")
st.title("Exploratory Data Visualizations for SDG Analysis")
st.write("Upload your data and explore interactive visualizations.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    #--------------------------------------------------------------------------------------------------------------------------------
    #Data wrangling
    full_merged_df = pd.read_csv(uploaded_file, parse_dates=["datetime"])  # Ensure 'datetime' is parsed correctly

    full_merged_df['week'] = full_merged_df['datetime'].dt.isocalendar().week
    weekly_velocity = full_merged_df.groupby(["week", "Velocity_Category"])["time_unit"].sum().reset_index()
    
    daily_velocity = full_merged_df.groupby(["week", "date", "Velocity_Category"])["time_unit"].sum().reset_index()
    avg_daily_velocity = daily_velocity.groupby("Velocity_Category")['time_unit'].mean().reset_index()
    
    daily_gate = full_merged_df.groupby(["week", "date","DGL"])["time_unit"].sum().reset_index()
    avg_daily_gate = daily_gate.groupby("DGL")['time_unit'].mean().reset_index()
    
    velocity_text = f"The daily average of velocity over 8ft/s over the entire time period is {avg_daily_velocity['time_unit'][0]:.2f} hours. The daily average of velocity under 8ft/s over the entire time period is {avg_daily_velocity['time_unit'][1]:.2f} hours."
    dgl_text = f"The daily average of the DGL Gate over the entire time period is {avg_daily_gate['time_unit'][0]:.2f} hours.  The daily average of the DGL Gate closed over the entire period is {avg_daily_gate['time_unit'][1]:.2f} hours."
    

    summary_stats_vel = (full_merged_df.groupby(["week", "date", "Velocity_Category"]).
        agg(
            total_velocity_duration = ("time_unit", "sum")
        )).reset_index()
    summary_stats_dgl = (full_merged_df.groupby(["week", "date", "DGL"]).
        agg(
            total_gate_duration = ("time_unit", "sum")
        )).reset_index()
    
    #--------------------------------------------------------------------------------------------------------------------------------
    # Create the graphs
    # Create velocity graph

    brush = alt.selection_interval(encodings=['x'], mark=alt.BrushConfig(stroke="cyan", strokeOpacity=1))
    base_vel = alt.Chart(summary_stats_vel, width=800, height=400).mark_bar(color="green").encode(
            x=alt.X("date:T", title="Velocity Category"),
            y=alt.Y("total_velocity_duration:Q", title="Hours"),
            color=alt.condition(brush, 'Velocity_Category:N', alt.value('lightgray')),
            tooltip=["date:T", "Velocity_Category:N", "total_velocity_duration:Q"],
    ).properties(
        title="Daily Velocity Over/Under 8 ft/s Duration Summary"
    )
    upper_vel = base_vel.mark_bar(width=alt.RelativeBandSize(0.7)).encode(
        alt.X('date:T').scale(domain=brush)
    )
    lower_vel = base_vel.properties(
        height=90
    ).add_params(brush)
    vel_bar_chart = upper_vel & lower_vel

    base_gate = alt.Chart(summary_stats_dgl, width=800, height=400).mark_bar(
        color="steelblue",
    ).encode(
        x=alt.X("date:T", title="Gate Status"), 
        y=alt.Y("total_gate_duration:Q", title="Hours"),
        color=alt.condition(brush, 'DGL:N', alt.value('lightgray')),
        tooltip=["date:T","DGL:N", "total_gate_duration:Q"]
    ).properties(
        title="Daily Gate Status Duration Summary"
    )
    
    upper_gate = base_gate.mark_bar(width=alt.RelativeBandSize(0.7)).encode(
        alt.X('date:T').scale(domain=brush)
    )
    lower_gate = base_gate.properties(
        height=90
    ).add_params(brush)
    
    gate_bar_chart = upper_gate & lower_gate
    combined_bar_charts = alt.vconcat(
        gate_bar_chart,
        vel_bar_chart
        )
#-----------------------------------------------------------------------------------------------------------------------------------
    df = full_merged_df.rename(
    columns={
        "GLC": "Flow (ft/s)",
        "datetime": "Datetime",
        "Velocity_Category": "Velocity Category",
        "consecutive_groups": "Consecutive Groups",
        "min_datetime": "Flow Duration Min Datetime",
        "max_datetime": "Flow Duration Max Datetime",
        "streak_duration": "Streak Duration (hrs)",
        "gate_min_datetime": "Gate Min Datetime",
        "gate_max_datetime": "Gate Max Datetime",
        "gate_count": "Gate Count",
        "gate_streak_duration": "Gate Streak Duration (hrs)",
        "time_unit": "Time Unit (hrs)",
    }
    )
    df = df.reset_index(drop=True)
    
    #-------------------------------------------------------------------------------------------------------------------------------
    # Markdown
    st.write("### Data Preview")
    st.dataframe(df.style.format(precision=2).set_table_styles(
        [{
            'selector': 'thead th',
            'props': [('background-color', '#4CAF50'), ('color', 'white'), ('text-align', 'center')]
        },
         {
            'selector': 'tbody tr:hover',
            'props': [('background-color', '#f5f5f5')]
        }]
    ), use_container_width=True)

    # Altair Visualization
    st.write("### Interactive Visualization")
    st.altair_chart(combined_bar_charts, use_container_width=True, theme=None)
    
    st.write("### Data Summary")
    st.write(velocity_text)
    st.write(dgl_text)
    

else:
    st.write("Please upload a CSV file to see the visualization.")
