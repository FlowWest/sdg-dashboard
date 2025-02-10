import psycopg2
import pandas as pd

gatef = {'name' : ['GrantLine','MiddleRiver','OldRiver'],
         'width' : [5,5,5],
         'C' : [0.8,0.8,0.8],
         'bottom_elev' :[-6,-5,-7],
         'ID': ['GLC','MID','OLD'],
         'station' : ['DGL','MHO','OLD'],
         "flow_op" : ["glc_flow_fish", "mid_flow_fish", "old_flow_fish"],
         "gate_status" : ["glc_gate_up", "mid_gate_up", "old_gate_up"]
        #  "flow_op" : ["GLC_FLOW_FISH", "MID_FLOW_FISH", "OLD_FLOW_FISH"],
        #  "gate_status" : ["GLC_GATE_UP", "MID_GATE_UP", "OLD_GATE_UP"]
         }

# stn_list = ['MID_GATEOP','GLC_GATEOP','OLD_GATEOP']
stn_list = ['mid_gateop', 'glc_gateop', 'old_gateop']
# elev_list =['MID_GATE_UP','MID_GATE_DOWN','GLC_GATE_UP','GLC_GATE_DOWN','OLD_GATE_UP','OLD_GATE_DOWN']
elev_list = ['mid_gate_up', 'mid_gate_down', 'glc_gate_up', 'glc_gate_down', 'old_gate_up', 'old_gate_down']
# flow_list = ['GLC_FLOW_FISH','MID_FLOW_FISH','MID_FLOW_GATE','OLD_FLOW_FISH','OLD_FLOW_GATE']
flow_list = ['glc_flow_fish', 'mid_flow_fish', 'mid_flow_gate', 'old_flow_fish', 'old_flow_gate']
# stn_name = ['MHO','DGL','OLD']
stn_name = ['mho', 'dgl', 'old']

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



def get_connection():
    return psycopg2.connect(
        host=db_secrets["host"],
        database=db_secrets["database"],
        user=db_secrets["user"],
        password=db_secrets["password"],
        port=db_secrets["port"]
    )

def fetch_data():
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        SELECT d.*, s.name 
        FROM dsm2 d
        JOIN scenarios s ON d.scenario_id = s.id
    """
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

db_data = fetch_data()
#db['node'] matches data['sdg']['gate_op']
columns = ['id', 'scenario_id', 'datetime','node', 'param','value', 'unit', 'updated_at', 'year','model_name']
db = pd.DataFrame(db_data, columns=columns)
cleaned_data = db.drop(["id", "scenario_id", "updated_at"], axis = 1)
cleaned_data_fpv1ma = cleaned_data[cleaned_data['model_name']=="FPV1Ma"]
start_date = '2016-05-01'
end_date = '2016-11-30'
sdg_stage = filter_data(cleaned_data_fpv1ma, 'node', elev_list, start_date, end_date)
sdg_flow = filter_data(cleaned_data_fpv1ma, 'node', flow_list, start_date, end_date)
sdg_gateop = filter_data(cleaned_data_fpv1ma, 'node', stn_list, start_date, end_date)
gate_names = {'mid_gateop': 'mho', 'glc_gateop': 'dgl', 'old_gateop': 'old'}
sdg_gateop = rename_gates(sdg_gateop, "node", gate_names)
    
# db_data = pd.DataFrame({
#     "datetime": db_data[3]

# })
# db_d = pd.DataFrame(db_data)
# cleaned_data = cleaned_data[cleaned_data['node'].isin(elev)].dropna()

# db_d[db_d['node']]
print(cleaned_data.columns)