import streamlit as st
import pandas as pd

from sqlalchemy import create_engine, text


@st.cache_resource
def get_db_connection():
    db_url = st.secrets["database"]["url"]
    return create_engine(db_url)


engine = get_db_connection()


def get_all_scenarios():
    q = """ SELECT name as "Scenario", comments as "Comments" from scenarios; """
    return pd.read_sql(q, engine)


def get_scenario_year_data(scenario, year):
    q = text(""" 
        SELECT * from dsm2 
        WHERE
            year = :year
        AND 
            scenario_id = (select id from scenarios where name = :scenario);
    """)
    params = {"year": year, "scenario": scenario}

    return pd.read_sql(q, engine, params=params)


# ---- utils
def calculate_vel(flow_fish, gate_ops, elev, width):
    d = flow_fish.merge(gate_ops, on=["datetime"])
    d = d[["datetime", "flow", "stage_up"]]
    d_xs = d.assign(xs=(d["stage_up"] - (elev)) * width)
    d_vel = d_xs.assign(vel=d_xs["flow"] / d_xs["xs"])
    return d_vel


def generate_scenario_year_data(data, widths=[5, 5, 5], elevs=[-6, -5, -7]):
    flow_op_nodes = ["glc_flow_fish", "mid_flow_fish", "old_flow_fish"]
    gate_up_nodes = ["glc_gate_up", "mid_gate_up", "old_gate_up"]
    names_in_order = ["glc", "mid", "old"]
    scenario_water_levels = data[
        (data["node"].isin(["dgl", "mho", "old"])) & (data["param"] == "stage")
    ]
    scenario_gate_operation = data[
        (data["node"].isin(["mid_gateop", "glc_gateop", "old_gateop"]))
    ]
    scenario_gate_up = data[(data.node.isin(gate_up_nodes)) & (data.unit == "FEET")]
    scenario_flow_data = data[data.node.isin(flow_op_nodes)]

    # generate velocity data
    out = []
    for fnode, gnode, w, e, n in zip(
        flow_op_nodes, gate_up_nodes, widths, elevs, names_in_order
    ):
        flow_fish = scenario_flow_data[scenario_flow_data.node == fnode].rename(
            columns={"value": "flow"}
        )
        gate_ops = scenario_gate_up[scenario_gate_up.node == gnode].rename(
            columns={"value": "stage_up"}
        )
        vels = calculate_vel(flow_fish, gate_ops, elev=e, width=w)
        vels["location"] = n
        out.append(vels)

    all_vels = pd.concat(out)

    return {
        "water_levels": scenario_water_levels,
        "flow": scenario_flow_data,
        "gate_operations": scenario_gate_operation,
        "vel": all_vels,
    }


def get_filter_nodes_for_gate(gate, dataset):
    return {
        "glc": {
            "water_levels": "dgl",
            "gate_operations": "glc_gateop",
            "flow": "glc_flow_fish",
        },
        "old": {
            "water_levels": "old",
            "gate_operations": "old_gateop",
            "flow": "old_flow_fish",
        },
        "mid": {
            "water_levels": "mid",
            "gate_operations": "mid_gateop",
            "flow": "mid_flow_fish",
        },
    }[gate][dataset]
