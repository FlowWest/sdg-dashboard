import pandas as pd
import altair as alt
from pandas import DataFrame
import numpy as np

location_gate = {
        "GLC":"Grantline",
        "MID":"MiddleRiver",
        "OLD":"OldRiver",
    }

def post_process_gateop(multi_model_data, model, gate, year=None, start_date=None, end_date=None):
    gate_data = multi_model_data[model][gate]['gate_operation_data']
    if year:
        gate_data["year"] = gate_data["datetime"].dt.year
        gate_data = gate_data[gate_data["year"] == year]
    # if start_date:
    #     gate_data['date'] = gate_data["datetime"].dt.date
    #     gate_data = gate_data[(gate_data["date"] >= start_date) & 
    #                           (gate_data["date"] <= end_date)]
    gate_data = gate_data.loc[
            gate_data['datetime'].dt.month.between(5, 11)
    ]
        # gate_data['datetime'].dt.month.between(5, 11)
    # ]
    gate_up_df = gate_data[["datetime", "value"]]
    gate_up_df["gate_status"] = gate_up_df['value']>=10
    gate_up_df['consecutive_groups'] = (gate_up_df['value'] != gate_up_df['value'].shift()).cumsum()
    gate_up_df['min_datetime'] = gate_up_df.groupby('consecutive_groups')['datetime'].transform('min')
    gate_up_df['max_datetime'] = gate_up_df.groupby('consecutive_groups')['datetime'].transform('max')
    consecutive_streaks = gate_up_df.groupby(['consecutive_groups', 'value', 'min_datetime', 'max_datetime']).size().reset_index(name='count')
    consecutive_streaks['streak_duration'] = consecutive_streaks['count'] * 15 / 60
    consecutive_streaks_clean = consecutive_streaks.drop(['value', 'consecutive_groups', 'max_datetime'], axis = 1)
    # print(consecutive_streaks_clean.head())
    merged_gate_df = pd.merge(gate_up_df, consecutive_streaks_clean,left_on="min_datetime", right_on="min_datetime")
    merged_gate_df = merged_gate_df.drop(['consecutive_groups', 'value'], axis=1)
    merged_gate_df = merged_gate_df.rename(columns={"min_datetime": "gate_min_datetime", 
                                                "max_datetime": "gate_max_datetime",
                                                "count": "gate_count",
                                                "streak_duration": "gate_streak_duration"})
    return merged_gate_df

def post_process_velocity(multi_model_data, model, gate, year=None, start_date=None, end_date=None):
    vel_zoom_df =multi_model_data[model][gate]["vel"]
    if year:
        vel_zoom_df["year"] = vel_zoom_df["datetime"].dt.year
        vel_zoom_df = vel_zoom_df[vel_zoom_df["year"] == year]
    vel_zoom_df = vel_zoom_df.loc[
            vel_zoom_df['datetime'].dt.month.between(5, 11)
    ]
    # if start_date:
    #     vel_zoom_df['date'] = vel_zoom_df["datetime"].dt.date
    #     vel_zoom_df = vel_zoom_df[(vel_zoom_df["date"] >= start_date) & 
    #                               (vel_zoom_df["date"] <= end_date)]
    vel_zoom_df['Velocity_Category'] = np.where(vel_zoom_df['value'] >= 8, "Over 8ft/s", "Under 8ft/s")
    #.shift shift value down and compare each value with the previous row; increase value when rows are different
    vel_zoom_df['consecutive_groups'] = (vel_zoom_df['Velocity_Category'] != vel_zoom_df['Velocity_Category'].shift()).cumsum()
    vel_zoom_df['min_datetime'] = vel_zoom_df.groupby('consecutive_groups')['datetime'].transform('min')
    vel_zoom_df['max_datetime'] = vel_zoom_df.groupby('consecutive_groups')['datetime'].transform('max')
    vel_zoom_df['date'] = vel_zoom_df['datetime'].dt.date.astype(str)
    consecutive_streaks_vel = vel_zoom_df.groupby(['consecutive_groups', 'Velocity_Category', 'min_datetime', 'max_datetime']).size().reset_index(name='count')
    consecutive_streaks_vel['streak_duration'] = consecutive_streaks_vel['count'] * 15 / 60

    consecutive_streaks_vel_clean = consecutive_streaks_vel.drop(['consecutive_groups', 'Velocity_Category', 'max_datetime'], axis=1)
    merged_vel_df = pd.merge(vel_zoom_df, consecutive_streaks_vel_clean,left_on="min_datetime", right_on="min_datetime")

    return merged_vel_df

def post_process_full_data(multi_model_data, model, gate, year=None, start_date=None, end_date=None):
    merged_gate_df = post_process_gateop(multi_model_data, model, gate, year, start_date, end_date)
    merged_vel_df = post_process_velocity(multi_model_data, model, gate, year, start_date, end_date)
    full_merged_df = pd.merge(merged_vel_df, merged_gate_df, left_on="datetime", right_on="datetime")
    full_merged_df['time_unit'] = 0.25

    full_merged_df['gate_status'] = np.where(full_merged_df['gate_status'], "Closed", "Open")
    full_merged_df['week'] = full_merged_df['datetime'].dt.isocalendar().week
    full_merged_df['gate'] = gate
    full_merged_df['model'] = model
    
    return full_merged_df

def post_process_hydro_data(multi_model_data, model, gate, year=None, start_date=None, end_date=None):
    hydro_df =multi_model_data[model][gate]["water_level_data"]
    if year:
        hydro_df["year"] = hydro_df["datetime"].dt.year
        hydro_df = hydro_df[hydro_df["year"] == year]
    hydro_df = hydro_df.loc[
            hydro_df['datetime'].dt.month.between(5, 11)
    ]
    hydro_df['time_unit'] = 0.25
    hydro_df = hydro_df.rename(columns={"value": "water_level"})
    hydro_df['week'] = hydro_df['datetime'].dt.isocalendar().week
    return(hydro_df)

def calc_avg_daily_vel(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of total amount of time velocity is above and below 8ft/s.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """
    daily_velocity = post_processed_data.groupby(["date", "Velocity_Category"])["time_unit"].sum().reset_index()
    avg_daily_velocity = pd.DataFrame(daily_velocity.groupby("Velocity_Category")['time_unit'].sum()/daily_velocity["date"].nunique()).reset_index()
    
    return avg_daily_velocity

def calc_avg_daily_gate(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of total amount of time gate is open and closed.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """
        
    daily_gate = post_processed_data.groupby(["date","gate_status"])["time_unit"].sum().reset_index()
    avg_daily_gate = pd.DataFrame(daily_gate.groupby("gate_status")['time_unit'].sum()/daily_gate["date"].nunique()).reset_index()

    return avg_daily_gate

def calc_avg_len_consec_vel(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of length of consecutive hours velocity is above and below 8ft/s.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """
    daily_velocity_stats = post_processed_data.groupby(["date", "Velocity_Category"]).agg(
        unique_consecutive_groups=("consecutive_groups", "nunique"),
        total_time=("time_unit", "sum")
    ).reset_index()

    daily_velocity_stats["daily_average_time_per_consecutive_group"] = (
        daily_velocity_stats["total_time"] / daily_velocity_stats["unique_consecutive_groups"]
    )
    daily_average_per_duration_per_velocity_over_period = daily_velocity_stats.groupby(["Velocity_Category"])['daily_average_time_per_consecutive_group'].mean().reset_index()
    
    return daily_average_per_duration_per_velocity_over_period

def calc_avg_len_consec_gate(post_processed_data: DataFrame) -> DataFrame:
    """
    Calculate daily average of length of consecutive hours gate is open or closed.

    Parameters:
    - post_processed_data (DataFrame): post processed dataframe.

    Returns:
    - DataFrame
    """    
    daily_gate_stats = post_processed_data.groupby(["date", "gate_status"]).agg(
        unique_gate_count=("gate_count", "nunique"),
        total_time=("time_unit", "sum")
    ).reset_index()

    daily_gate_stats["daily_average_time_per_consecutive_gate"] = (
        daily_gate_stats["total_time"] / daily_gate_stats["unique_gate_count"]
    )
    daily_average_per_duration_per_gate_over_period = daily_gate_stats.groupby(["gate_status"])['daily_average_time_per_consecutive_gate'].mean().reset_index()
    
    return daily_average_per_duration_per_gate_over_period

def generate_velocity_gate_charts(full_merged_df, legend=None):
    """
    Generate bar charts for daily velocity and gate status summaries.

    Parameters:
    full_merged_df (pd.DataFrame): DataFrame containing the merged data with the necessary columns.

    Returns:
    alt.Chart: Combined Altair charts for velocity and gate status duration summaries.
    """
    # Compute summary statistics
    summary_stats_vel = (full_merged_df.groupby(["week", "date", "Velocity_Category"]).
            agg(
                total_velocity_duration=("time_unit", "sum")
            )).reset_index()

    summary_stats_dgl = (full_merged_df.groupby(["week", "date", "gate_status"]).
        agg(
            total_gate_duration=("time_unit", "sum")
        )).reset_index()

    # Extract unique gate identifier
    gate = full_merged_df['gate'].unique()[0]
    model = full_merged_df['model'].unique()[0]

    # Define color palette
    color_palette = {
        "Velocity_Category": {"Over 8ft/s": "#a6cee3", "Under 8ft/s": "#1f78b4"},  # Blues
        "gate_status": {"Closed": "#b2df8a", "Open": "#33a02c"}  # Greens
    }

    # Define brush for selection
    brush = alt.selection_interval(encodings=['x'], mark=alt.BrushConfig(stroke="cyan", strokeOpacity=1))

    # Velocity bar chart
    if legend:
        base_vel = alt.Chart(summary_stats_vel, width=650, height=300).mark_bar().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("total_velocity_duration:Q", title="Hours"),
            color=alt.condition(
                brush,
                alt.Color(
                    'Velocity_Category:N',
                    title="Velocity Category",
                    scale=alt.Scale(
                        domain=color_palette["Velocity_Category"].keys(),
                        range=color_palette["Velocity_Category"].values()
                    )
                ),
                alt.value('lightgray')  # Gray for unselected
            ),
            tooltip=["date:T", "Velocity_Category:N", "total_velocity_duration:Q"]
        ).properties(
            title=f"Daily Velocity at {location_gate[gate]} Over/Under 8 ft/s Duration Summary"
        )
    else:
        base_vel = alt.Chart(summary_stats_vel, width=600, height=300).mark_bar().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("total_velocity_duration:Q", title="Hours"),
            color=alt.condition(
                brush,
                alt.Color(
                    'Velocity_Category:N',
                    title="Velocity Category",
                    scale=alt.Scale(
                        domain=color_palette["Velocity_Category"].keys(),
                        range=color_palette["Velocity_Category"].values()
                    ),
                    legend=None
                ),
                alt.value('lightgray')  # Gray for unselected
            ),
            tooltip=["date:T", "Velocity_Category:N", "total_velocity_duration:Q"]
        ).properties(
            title=f"Daily Velocity at {location_gate[gate]} Over/Under 8 ft/s Duration Summary"
        )

    upper_vel = base_vel.mark_bar(width=alt.RelativeBandSize(0.7), stroke='grey', strokeWidth=0.5).encode(
        alt.X('date:T', title="Date").scale(domain=brush)
    )

    lower_vel = base_vel.properties(
        height=90
    ).add_params(brush)

    vel_bar_chart = upper_vel & lower_vel

    # Gate bar chart
    if legend:
        base_gate = alt.Chart(summary_stats_dgl, width=600, height=300).mark_bar().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("total_gate_duration:Q", title="Hours"),
            color=alt.condition(
                brush,
                alt.Color(
                    'gate_status:N',
                    title="Gate Status",
                    scale=alt.Scale(
                        domain=color_palette["gate_status"].keys(),
                        range=color_palette["gate_status"].values()
                    )
                ),
                alt.value('lightgray')  # Gray for unselected
            ),
            tooltip=["date:T", "gate_status:N", "total_gate_duration:Q"]
        ).properties(
            title=f"Daily {gate} Gate Status Duration Summary"
        )
    else:
                base_gate = alt.Chart(summary_stats_dgl, width=600, height=300).mark_bar().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("total_gate_duration:Q", title="Hours"),
            color=alt.condition(
                brush,
                alt.Color(
                    'gate_status:N',
                    title="Gate Status",
                    scale=alt.Scale(
                        domain=color_palette["gate_status"].keys(),
                        range=color_palette["gate_status"].values()
                    ),
                    legend=None
                ),
                alt.value('lightgray')  # Gray for unselected
            ),
            tooltip=["date:T", "gate_status:N", "total_gate_duration:Q"]
        ).properties(
            title=f"Daily {gate} Gate Status Duration Summary"
        )

    upper_gate = base_gate.mark_bar(width=alt.RelativeBandSize(0.7), stroke='grey', strokeWidth=0.5).encode(
        alt.X('date:T', title="Date").scale(domain=brush)
    )

    lower_gate = base_gate.properties(
        height=90
    ).add_params(brush)

    gate_bar_chart = upper_gate & lower_gate

    # Combine charts
    combined_bar_charts = alt.vconcat(
        gate_bar_chart,
        vel_bar_chart
    ).resolve_scale(color='independent').properties(title= f"Model: {model}")

    return combined_bar_charts

def generate_zoomed_velocity_charts(filtered_merged_df):
    color_palette = {
         "Velocity_Category": {"Over 8ft/s": "#a6cee3", "Under 8ft/s": "#1f78b4"},  # Blues
         "gate_status": {"Closed": "#b2df8a", "Open": "#33a02c"}  # Greens
         }
    gate = filtered_merged_df['gate'].unique()[0]
    model = filtered_merged_df['model'].unique()[0]

    interval = alt.selection_interval(encodings=['x'],
                                      mark=alt.BrushConfig(fill='blue')
                                      )
    base = alt.Chart(filtered_merged_df).mark_line(color=color_palette["Velocity_Category"]["Under 8ft/s"]).encode(
        x=alt.X(
            'yearmonthdatehoursminutes(datetime):T', 
            title='Datetime', 
            axis=alt.Axis(format='%b %d, %Y', labelAngle=-45, title='Date')
        ),
        y=alt.Y('velocity:Q', title='Velocity (ft/s)'),
    ).add_params(interval).properties(
        title=f"Fish Passage Velocity and {gate} Gate Status Zoomed",
        height=300
    )

    closed_gates = filtered_merged_df[['gate_min_datetime', 'gate_max_datetime', 'gate_status']].drop_duplicates().reset_index(drop=True)
    area_gate_true = alt.Chart(closed_gates).mark_rect(
        color='orange'
    ).encode(
        x='gate_min_datetime:T',
        x2='gate_max_datetime:T',
        opacity=alt.value(0.2),
        color=alt.value(color_palette["gate_status"]["Closed"]),
        # color=alt.condition(interval, alt.value('orange'), alt.value('lightgray'))
    ).transform_filter(
        alt.datum.gate_status == "Closed"
    )
    yrule = alt.Chart(filtered_merged_df).mark_rule(color = "red", strokeDash=[12, 6], size=1.5).encode(
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
    rules = alt.Chart(filtered_merged_df).transform_calculate(
        FlowVelocityDuration = "'Flow ' + datum.Velocity_Category + ' duration is ' + datum.streak_duration + ' hours'",
        GateStatusDuration = "'Gate ' + datum.gate_status + ' duration is ' + datum.gate_streak_duration + ' hours'"
    ).mark_rule(color="gray").encode(
        x="datetime:T",
        opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
        tooltip=[alt.Tooltip('yearmonthdatehoursminutes(datetime):T', title='Datetime'),
                 alt.Tooltip('velocity:Q', title= "Velocity (ft/s)", format=".2f"),
                 alt.Tooltip('FlowVelocityDuration:N', title="Flow Velocity Duration"),
                 alt.Tooltip('GateStatusDuration:N', title="Gate Status Duration")
                 ],
    ).add_params(nearest)

    vel_text = alt.Chart(filtered_merged_df).mark_text(align='right').encode(
        y=alt.Y('stat:N', axis=None),
        text=alt.Text('summary:N')
    ).transform_filter(
        interval
    ).transform_aggregate(
        max_velocity='max(velocity)',  
        min_velocity='min(velocity)',
        avg_velocity='mean(velocity)'
    ).transform_fold(
        ['max_velocity', 'min_velocity', 'avg_velocity'],  # Separate each statistic
        as_=['stat', 'value']
    ).transform_calculate(
        summary='datum.stat + ": " + format(datum.value, ".2f")'
    )
    velocity = vel_text.encode(text='summary:N').properties(
        title=alt.Title(text='Flow Velocity Summary', align='center')
    )

    vel_duration_text = alt.Chart(filtered_merged_df).mark_text(align='right').encode(
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

    gate_duration_text = alt.Chart(filtered_merged_df).mark_text(align='right').encode(
        y=alt.Y('gate_status:N', axis=None),
        text=alt.Text('summary:N')
    ).transform_filter(
        interval
    ).transform_aggregate(
        total_time='sum(time_unit)',
        groupby=["gate_status"]
    ).transform_fold(
        ['total_time'],  # Separate each statistic
        as_=['stat', 'value']
    ).transform_calculate(
        summary='datum.gate_status + ": " + format(datum.total_time, ".2f") + " hours"'
    )
    gate_duration = gate_duration_text.encode(text='summary:N').properties(
        title=alt.Title(text='Gate Status Duration Summary', align='center')
    )

    layered_chart = alt.layer(base, points, yrule, rules, area_gate_true).properties(
        width=600,  
        height=400
    )
    velocity = velocity.properties(width=200, height=100)
    # daily_velocity = daily_velocity.properties(width=200, height=100)
    velocity_duration = velocity_duration.properties(width=200, height=100)
    gate_duration = gate_duration.properties(width=200, height=100)

    text_summary = alt.hconcat(velocity, velocity_duration, gate_duration)
    
    combined_chart = alt.vconcat(
        layered_chart,
        # velocity
        text_summary
    ).properties(title= f"Model: {model}")

    return combined_chart

def generate_water_level_chart(filtered_hydro_df, filtered_merged_df):
    color_palette = {
        "Velocity_Category": {"Over 8ft/s": "#a6cee3", "Under 8ft/s": "#1f78b4"},  # Blues
        "gate_status": {"Closed": "#b2df8a", "Open": "#33a02c"}  # Greens
    }
    gate = filtered_hydro_df['gate'].unique()[0]
    model = filtered_hydro_df['scenario'].unique()[0]
    interval = alt.selection_interval(encodings=['x'],
                                  mark=alt.BrushConfig(fill='blue'))
    base_elevation = alt.Chart(filtered_hydro_df).mark_line().encode(
            x=alt.X('yearmonthdatehoursminutes(datetime):T', title='Date', axis=alt.Axis(format='%b %d, %Y', labelAngle=-45)),
            y=alt.Y('water_level:Q', title='Feet'),
            # color='scenario:N'
        ).transform_fold(
            ["scenario"],  # Columns to be "melted" into a long format
            as_=['model', 'value']  # New column names
        ).add_params(
            interval
        ).properties(
            title=f"Summary of Minimum Stage At {gate}"
        )
    yrule_wl = alt.Chart(filtered_hydro_df).mark_rule(color = "purple", strokeDash=[12, 6], size=1.5).encode(
                y=alt.datum(2.3)
        )
    yrule_wl_text = alt.Chart(filtered_hydro_df).mark_text(
            text="Water Level Compliance",
            align="left",
            baseline="bottom",
            fontSize=12,
            color="grey",
            dx=5  # Offset text slightly to the right of the rule
        ).encode(
            y=alt.datum(2.3)  # Same y position as the rule
        )
    nearest_elev = alt.selection_point(nearest=True, on="pointerover",
                                      fields=["datetime"], empty=False)
    points_elev = base_elevation.mark_point().encode(
            opacity=alt.condition(nearest_elev, alt.value(1), alt.value(0))
        )
    rules_elev = alt.Chart(filtered_hydro_df).mark_rule(color="gray", opacity=0).encode(
            x="datetime:T",
            opacity=alt.condition(nearest_elev, alt.value(0.3), alt.value(0)),
        ).add_params(nearest_elev)
    when_near = alt.when(nearest_elev)
    text = base_elevation.mark_text(
            align="left", dx=5, dy=-5
        ).transform_calculate(
            label='format(datum.water_level, ".2f") + " feet"'
        ).encode(
            text=when_near.then("label:N").otherwise(alt.value(" "))
        )
    average_scenario_stage = alt.Chart(filtered_hydro_df).mark_text(align='right').encode(
                y=alt.Y('stat:N', axis=None),
                text=alt.Text('summary:N')
            ).transform_filter(
                interval
            ).transform_aggregate(
                average_stage='mean(water_level)'
            ).transform_fold(
                ['average_stage'],  # Separate each statistic
                as_=['stat', 'value']
            ).transform_calculate(
                summary='format(datum.average_stage, ".2f") + " feet"'
            )
    avg_stage = average_scenario_stage.encode(text='summary:N').properties(
                title=alt.Title(text=f'Average {model} Minimum Stage', align='center')
        )
    scenario_duration_below_wl = alt.Chart(filtered_hydro_df).mark_text(align='right').encode(
            y=alt.Y('wl_stat:N', axis=None),
                text=alt.Text('below_wl:N')
            # ).transform_filter(
            #     interval
            ).transform_filter(
                "datum.FPV1Ma < 2.3"
            ).transform_aggregate(
                total_time_below_wl='sum(time_unit)'
            ).transform_calculate(
                below_wl='format(datum.total_time_below_wl, ".2f") + " hour"'
            ).properties(
                title=alt.Title(text='Scenario Below Water Level Compliance', align='center')
            )
    closed_gates = filtered_merged_df[['gate_min_datetime', 'gate_max_datetime', 'gate_status']].drop_duplicates().reset_index(drop=True)
    area_gate_true = alt.Chart(closed_gates).mark_rect(
        color='orange'
    ).encode(
        x='gate_min_datetime:T',
        x2='gate_max_datetime:T',
        opacity=alt.value(0.2),
        color=alt.value(color_palette["gate_status"]["Closed"]),
        # color=alt.condition(interval, alt.value('orange'), alt.value('lightgray'))
    ).transform_filter(
        alt.datum.gate_status == "Closed"
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
    scenario_duration_below_wl = scenario_duration_below_wl.properties(width=200, height=100)
    # modeled_historic_below_wl = modeled_historic_below_wl.properties(width=200, height=100)
        # (base, points, yrule, rules, area_dgl_true
    min_stage_chart = alt.layer(base_elevation, 
                                  points_elev, 
                                  yrule_wl, 
                                  yrule_wl_text,
                                  rules_elev,
                                  area_gate_true,
                                  text
        ).properties(width=700, height=400, title= f"Model: {model}")
    return min_stage_chart