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
    
    min_date = min(daily_velocity['date'])
    max_date = max(daily_velocity['date']) 

    summary_stats_title = f"Summary Stats from {min_date} to {max_date}."

    summary_data = {
        "Metric": [
            f"Daily average velocity {avg_daily_velocity['Velocity_Category'][0]}",
            f"Daily average velocity {avg_daily_velocity['Velocity_Category'][1]}",
            f"Daily average DGL Gate {avg_daily_gate['DGL'][0]}",
            f"Daily average DGL Gate {avg_daily_gate['DGL'][1]}"
        ],
        "Hours": [
            f"{avg_daily_velocity['time_unit'][0]:.2f}",
            f"{avg_daily_velocity['time_unit'][1]:.2f}",
            f"{avg_daily_gate['time_unit'][0]:.2f}",
            f"{avg_daily_gate['time_unit'][1]:.2f}"
        ]
    }

    # Create a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # velocity_text = f"The daily average of velocity over 8ft/s over the entire time period is {avg_daily_velocity['time_unit'][0]:.2f} hours. The daily average of velocity under 8ft/s over the entire time period is {avg_daily_velocity['time_unit'][1]:.2f} hours."
    # dgl_text = f"The daily average of the DGL Gate over the entire time period is {avg_daily_gate['time_unit'][0]:.2f} hours.  The daily average of the DGL Gate closed over the entire period is {avg_daily_gate['time_unit'][1]:.2f} hours."
    

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
    # Vis 1
    # velocity graph
    brush = alt.selection_interval(encodings=['x'], mark=alt.BrushConfig(stroke="cyan", strokeOpacity=1))
    base_vel = alt.Chart(summary_stats_vel, width=800, height=300).mark_bar(color="green").encode(
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
    
    #gate graph
    base_gate = alt.Chart(summary_stats_dgl, width=800, height=300).mark_bar(
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
    
    # Vis 2
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
    st.write('#')
    st.write("### Visualization 1: Daily Gate Status Duration vs Daily Velocity Flow Duration")
    st.altair_chart(combined_bar_charts, use_container_width=True, theme=None)
    st.write('###')
    st.write("### Visualization 2: Flow Velocity and Gate Status Zoomed in By Week")
    drop_down_week = full_merged_df['week'].unique().tolist()
    selected_week = st.selectbox('Select Week:', drop_down_week)
    
    # Filter the data manually based on user input
    filtered_df = full_merged_df[full_merged_df['week'] == selected_week]
    #-------------------------------------------------------------------------------------------------------
    # Create an Altair chart using the filtered data
    interval = alt.selection_interval(encodings=['x'],
                                      mark=alt.BrushConfig(fill='blue')
                                      )
    base = alt.Chart(filtered_df).mark_line(color = "darkgreen").encode(
        x=alt.X('yearmonthdatehoursminutes(datetime):T', title='Datetime', axis=alt.Axis(format='%b %d, %Y', 
                                                                                         labelAngle=-45,
                                                                                         title='Date')),
        y=alt.Y('GLC:Q', title='Velocity (ft/s)'),
        # color=alt.when(interval).then(alt.value("darkgreen")).otherwise(alt.value("lightgray"))
        # color=alt.condition(interval, alt.value("darkgreen"), alt.value("lightgray"))
    ).add_selection(
        interval,
    ).properties(
        title="Flow Velocity and Gate Status Zoomed",
        height = 300
    )

    closed_gates = filtered_df[['gate_min_datetime', 'gate_max_datetime', 'DGL']].drop_duplicates().reset_index(drop=True)
    area_dgl_true = alt.Chart(closed_gates).mark_rect(
        color='orange'
    ).encode(
        x='gate_min_datetime:T',
        x2='gate_max_datetime:T',
        opacity=alt.value(0.2),
        # color=alt.condition(interval, alt.value('orange'), alt.value('lightgray'))
    ).transform_filter(
        alt.datum.DGL == "Closed"
    )
    yrule = alt.Chart(filtered_df).mark_rule(color = "red", strokeDash=[12, 6], size=1.5).encode(
            y=alt.datum(8)
    ).properties(
        width=300,
        height=300
    )

    nearest = alt.selection_point(nearest=True, on="pointerover",
                                  fields=["datetime"], empty=False)
    points = base.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # # Draw a rule at the location of the selection
    rules = alt.Chart(filtered_df).transform_calculate(
        FlowVelocityDuration = "'Flow ' + datum.Velocity_Category + ' duration is ' + datum.streak_duration + ' hours'",
        GateStatusDuration = "'Gate ' + datum.DGL + ' duration is ' + datum.gate_streak_duration + ' hours'"
    ).mark_rule(color="gray").encode(
        x="datetime:T",
        opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
        tooltip=[alt.Tooltip('yearmonthdatehoursminutes(datetime):T', title='Datetime'),
                 alt.Tooltip('GLC:Q', title= "Velocity (ft/s)", format=".2f"),
                 alt.Tooltip('FlowVelocityDuration:N', title="Flow Velocity Duration"),
                 alt.Tooltip('GateStatusDuration:N', title="Gate Status Duration")
                 ],
    ).add_params(nearest)

    vel_text = alt.Chart(filtered_df).mark_text(align='right').encode(
        y=alt.Y('stat:N', axis=None),
        text=alt.Text('summary:N')
    ).transform_filter(
        interval
    ).transform_aggregate(
        max_velocity='max(GLC)',  
        min_velocity='min(GLC)',
        avg_velocity='mean(GLC)'
    ).transform_fold(
        ['max_velocity', 'min_velocity', 'avg_velocity'],  # Separate each statistic
        as_=['stat', 'value']
    ).transform_calculate(
        summary='datum.stat + ": " + format(datum.value, ".2f")'
    )
    velocity = vel_text.encode(text='summary:N').properties(
        title=alt.Title(text='Flow Velocity Summary', align='center')
    )

    vel_duration_text = alt.Chart(filtered_df).mark_text(align='right').encode(
        y=alt.Y('Velocity_Category:N', axis=None),
        text=alt.Text('summary:N')
    ).transform_filter(
        interval
    ).transform_aggregate(
        total_time='sum(time_unit)',
        groupby=["Velocity_Category"]
    ).transform_fold(
        ['total_time'],  # Separate each statistic
        as_=['stat', 'value']
    ).transform_calculate(
        summary='datum.Velocity_Category + ": " + format(datum.total_time, ".2f") + " hours"'
    )
    velocity_duration = vel_duration_text.encode(text='summary:N').properties(
        title=alt.Title(text='Velocity Duration Summary', align='center')
    )

    gate_duration_text = alt.Chart(filtered_df).mark_text(align='right').encode(
        y=alt.Y('DGL:N', axis=None),
        text=alt.Text('summary:N')
    ).transform_filter(
        interval
    ).transform_aggregate(
        total_time='sum(time_unit)',
        groupby=["DGL"]
    ).transform_fold(
        ['total_time'],  # Separate each statistic
        as_=['stat', 'value']
    ).transform_calculate(
        summary='datum.DGL + ": " + format(datum.total_time, ".2f") + " hours"'
    )
    gate_duration = gate_duration_text.encode(text='summary:N').properties(
        title=alt.Title(text='Gate Status Duration Summary', align='center')
    )

    layered_chart = alt.layer(base, points, yrule, rules, area_dgl_true).properties(
        width=700,  
        height=400
    )
    velocity = velocity.properties(width=200, height=100)
    velocity_duration = velocity_duration.properties(width=200, height=100)
    gate_duration = gate_duration.properties(width=200, height=100)

    text_summary = alt.vconcat(velocity, velocity_duration, gate_duration)

    combined_chart = alt.hconcat(
        layered_chart,
        text_summary
    )

    # Display the chart in Streamlit
    st.altair_chart(combined_chart, use_container_width=False)
    
    st.write("### Data Summary")
    st.write(summary_stats_title)
    st.table(summary_df)    

else:
    st.write("Please upload a CSV file to see the visualization.")
