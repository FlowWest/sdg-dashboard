from backend.dsm2_reader import read_scenario_dir
from pandas import DataFrame, Series
import pandas as pd
from data_config import gatef, elev_list, flow_list, stn_name, stn_list
import numpy as np
from typing import Union, Optional, List, Dict
import os

def calc_vel(flow: float|Series, stage_up: float|Series, bottom_elev: float, width: float) -> float|Series:
    """
    Calculate velocity based on flow and cross-sectional area.

    Parameters:
    - flow (float or Series): Flow rate in cubic feet per second (cfs).
    - stage_up (float or Series): Upstream stage height in feet.
    - bottom_elev (float): Bottom elevation in feet.
    - width (float): Channel width in feet.

    Returns:
    - float or Series: Velocity in feet per second.
    """
    xs = (stage_up-bottom_elev)*width
    vel = flow/xs
    return vel

def filter_data(data: DataFrame, column: str, values: List, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Filter data based on a column and a list of values.
    Parameters:
    - data (DataFrame): Input data.
    - column (str): Column to filter on.
    - values (list): List of values to match.
    - start_date (str or None): Start date in YYYY-MM-DD format.
    - end_date (str or None): End date in YYYY-MM-DD format.

    Returns:
    - DataFrame: Filtered data.
    """
    if start_date:
        data = data[data[column].isin(values)].dropna()
        data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
    else:
        data = data[data[column].isin(values)].dropna()
    return data

def rename_gates(data: DataFrame, column: str, rename_map: Dict) -> DataFrame:
    """
    Rename values in a column in the DataFrame.
    Parameters:
    - data (DataFrame): Input data.
    - column (str): Column to rename values in.
    - rename_map (dict): Mapping of old values to new values.

    Returns:
    - DataFrame: Data with renamed column values.
    """
    data[column] = data[column].replace(rename_map)
    return data

def set_datetime_index(data: DataFrame) -> Series:
    """
    Set the datetime column as the index and return the 'value' column.

    Parameters:
    - data (DataFrame): Input data.

    Returns:
    - Series: 'value' column with datetime index.
    """
    data.set_index('datetime', inplace=True)
    data = data['value']
    return(data)

def parse_dss_filename(file_path: str) -> str:
    """
    Extract the last value from a DSS file path, excluding the extension.

    Parameters:
    - file_path (str): Full path to the DSS file.

    Returns:
    - str: The model name extracted from DSS file.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    model_name = file_name.split('_')[0]
    return model_name

def prepare_full_data(sdg_flow: DataFrame, 
                      sdg_stage: DataFrame, 
                      sdg_gateop: DataFrame, 
                      hydro_wl: DataFrame, 
                      gatef: Dict, 
                      model: str) -> Dict:
    """
    Prepare gate data dictionary from input datasets.
    
    Parameters:
    - sdg_flow: DataFrame with flow data.
    - sdg_stage: DataFrame with stage data.
    - sdg_gatop: DataFrame with gate operation data.
    - hydro_wl: Dataframe with water level data.
    - gatef: Dictionary containing data configuration.
    - model: model name

    Returns:
    - Dictionary containing gate-specific data.
    """
    full_data = {}
    for i, gate_name in enumerate(gatef['name']):
        flow_data = filter_data(sdg_flow, 'gate_op', [gatef["flow_op"][i]])
        gate_data = filter_data(sdg_stage, 'gate_op', [gatef['gate_status'][i]])
        gate_data = gate_data[gate_data['unit'] == "FEET"]
        gateop_data = sdg_gateop[sdg_gateop['gate_op']==gatef['station'][i]]
        hydro_wl_data = hydro_wl[hydro_wl['gate']==gatef['station'][i]]
        full_data[gatef["ID"][i]] = {
            'name': gatef['name'][i],
            'bottom_elev': gatef['bottom_elev'][i],
            'width': gatef['width'][i],
            'flow_data': flow_data, #sc2_flow
            'gate_data': gate_data, #sc2_gate
            'gate_operation_data': gateop_data, #sc2_gateop
            "water_level_data": hydro_wl_data, #sc2_wl
            "model": model
        }
    return full_data

def generate_full_model_data(data: Dict, 
                             path: str, 
                             gatef: Dict, 
                             elev_list: List, 
                             flow_list: List, 
                             stn_name: List, 
                             stn_list: List, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Dict:
    """
    Generate model data for gates based on input data and configurations.

    Parameters:
    - data (dict): Dictionary containing datasets.
    - path (str): Key to access the datasets in the dictionary.
    - gatef (Dictionary): Gate configuration data found in data_config.py.
    - elev_list (list): List of elevation values.
    - flow_list (list): List of flow values.
    - stn_name (list): Station names.
    - stn_list (list): Station IDs.
    - start_date (str or None): Start date in YYYY-MM-DD format.
    - end_date (str or None): End date in YYYY-MM-DD format.

    Returns:
    - dict: Dictionary containing processed model data.
    """
    sdg = data[path]['sdg']
    hydro = data[path]['hydro']
    model = parse_dss_filename(path)

    sdg_stage = filter_data(sdg, 'gate_op', elev_list, start_date, end_date)
    sdg_flow = filter_data(sdg, 'gate_op', flow_list, start_date, end_date)
    sdg_gateop = filter_data(sdg, 'gate_op', stn_list, start_date, end_date)
    gate_names = {'MID_GATEOP': 'MHO', 'GLC_GATEOP': 'DGL', 'OLD_GATEOP': 'OLD'}
    sdg_gateop = rename_gates(sdg_gateop, "gate_op", gate_names)
    
    hydro_wl = hydro[hydro['parameter']=="STAGE"]
    hydro_wl = filter_data(hydro_wl, 'gate', stn_name, start_date, end_date)

    full_data = prepare_full_data(sdg_flow, sdg_stage, sdg_gateop, hydro_wl, gatef, model)
    
    sdg_flow_GLC_FLOW_FISH = set_datetime_index(full_data['GLC']['flow_data'])
    sdg_flow_GLC_GATE_UP = set_datetime_index(full_data['GLC']['gate_data'])
    glc_bottom_elev = full_data['GLC']['bottom_elev']
    glc_width = full_data['GLC']['width']


    sdg_flow_MID_FLOW_FISH = set_datetime_index(full_data['MID']['flow_data'])
    sdg_flow_MID_GATE_UP = set_datetime_index(full_data['MID']['gate_data'])
    mid_bottom_elev = full_data['MID']['bottom_elev']
    mid_width = full_data['MID']['width']
    
    sdg_flow_OLD_FLOW_FISH = set_datetime_index(full_data['OLD']['flow_data'])
    sdg_flow_OLD_GATE_UP = set_datetime_index(full_data['OLD']['gate_data'])
    old_bottom_elev = full_data['OLD']['bottom_elev']
    old_width = full_data['OLD']['width']
    
    full_data["GLC"]['vel'] = pd.DataFrame(calc_vel(sdg_flow_GLC_FLOW_FISH,sdg_flow_GLC_GATE_UP,glc_bottom_elev, glc_width))
    full_data["MID"]['vel'] = pd.DataFrame(calc_vel(sdg_flow_MID_FLOW_FISH,sdg_flow_MID_GATE_UP,mid_bottom_elev, mid_width))
    full_data["OLD"]['vel'] = pd.DataFrame(calc_vel(sdg_flow_OLD_FLOW_FISH,sdg_flow_OLD_GATE_UP,old_bottom_elev, old_width))

    for key in ["GLC", "MID", "OLD"]:
        full_data[key]['vel']['datetime'] = full_data[key]['vel'].index
        full_data[key]['vel'] = full_data[key]['vel'].reset_index(drop=True)
        full_data[key]['vel'] = full_data[key]['vel'][["datetime", "value"]]
    
    return full_data

def post_process_gateop(model_data: Dict, gate: str) -> DataFrame:
    """
    Post-process gate operation data to identify consecutive groups and streaks.

    Parameters:
    - model_data (dict): Model data dictionary.
    - gate (str): Gate identifier.

    Returns:
    - DataFrame: Processed gate operation data.
    """
    gate_up_df = model_data[gate]['gate_operation_data'][["datetime", "value"]]
    gate_up_df["gate_status"] = gate_up_df['value']>=10
    gate_up_df['consecutive_groups'] = (gate_up_df['value'] != gate_up_df['value'].shift()).cumsum()
    gate_up_df['min_datetime'] = gate_up_df.groupby('consecutive_groups')['datetime'].transform('min')
    gate_up_df['max_datetime'] = gate_up_df.groupby('consecutive_groups')['datetime'].transform('max')
    consecutive_streaks = gate_up_df.groupby(['consecutive_groups', 'value', 'min_datetime', 'max_datetime']).size().reset_index(name='count')
    consecutive_streaks['streak_duration'] = consecutive_streaks['count'] * 15 / 60
    consecutive_streaks_clean = consecutive_streaks.drop(['value', 'consecutive_groups', 'max_datetime'], axis = 1)
    merged_gate_df = pd.merge(gate_up_df, consecutive_streaks_clean,left_on="min_datetime", right_on="min_datetime")
    merged_gate_df = merged_gate_df.drop(['consecutive_groups', 'value'], axis=1)
    merged_gate_df = merged_gate_df.rename(columns={"min_datetime": "gate_min_datetime", 
                                                "max_datetime": "gate_max_datetime",
                                                "count": "gate_count",
                                                "streak_duration": "gate_streak_duration"})
    return merged_gate_df

def post_process_velocity(model_data: Dict, gate: str) -> DataFrame:
    """
    Post-process velocity data to categorize and identify consecutive streaks.

    Parameters:
    - model_data (dict): Model data dictionary.
    - gate (str): Gate identifier.

    Returns:
    - DataFrame: Processed velocity data.
    """
    vel_zoom_df =model_data[gate]["vel"]
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

def post_process_full_data(model_data: Dict, gate: str) -> DataFrame:
    """
    Combine processed gate operation and velocity data into a single DataFrame.

    Parameters:
    - model_data (dict): Model data dictionary.
    - gate (str): Gate identifier.

    Returns:
    - DataFrame: Combined processed data.
    """
    merged_gate_df = post_process_gateop(model_data, gate)
    merged_vel_df = post_process_velocity(model_data, gate)
    full_merged_df = pd.merge(merged_vel_df, merged_gate_df, left_on="datetime", right_on="datetime")
    full_merged_df['time_unit'] = 0.25
    full_merged_df['gate_status'] = np.where(full_merged_df['gate_status'], "Closed", "Open")
    full_merged_df['week'] = full_merged_df['datetime'].dt.isocalendar().week
    full_merged_df['gate'] = gate
    full_merged_df['model'] = model_data[gate]["model"]
    
    return full_merged_df

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
#----------------------------------------------------------------------------------------------------
data = read_scenario_dir("C:/Users/Inigo/Projects/sdg-dashboard/data", v7_filter="_7")
start_zoom = '2016-05-01'
end_zoom = '2016-11-30'
# path = 'C:\\Users\\Inigo\\Projects\\sdg-dashboard\\data\\output\\FPV1Ma_hydro_7.dss'
fpv1ma_hydro_full_data = generate_full_model_data(data, 
                 'C:\\Users\\Inigo\\Projects\\sdg-dashboard\\data\\output\\FPV1Ma_hydro_7.dss', 
                 gatef,
                 elev_list,
                 flow_list,
                 stn_name,
                 stn_list,
                 start_zoom,
                 end_zoom
                 )

fpv2ma_hydro_full_data = generate_full_model_data(data,
                'C:\\Users\\Inigo\\Projects\\sdg-dashboard\\data\\output\\FPV2Ma_hydro_7.dss',
                gatef,
                elev_list,
                flow_list,
                stn_name,
                stn_list,
                start_zoom,
                end_zoom)

full_data = post_process_full_data(fpv2ma_hydro_full_data, "GLC")
print(full_data.head())