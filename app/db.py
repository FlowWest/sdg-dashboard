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


def get_gateop_elevations_for_scenario(scenario, year=None, backend=None):
    if year:
        q = text(""" 
            SELECT * from dsm2 
            WHERE
                node in ('mid_gate_up','mid_gate_down','glc_gate_up','glc_gate_down','old_gate_up','old_gate_down')
            AND 
                year = :year
            AND 
                scenario_id = (select id from scenarios where name = :scenario);
        """)
        params = {"year": year, "scenario": scenario}
    else:
        q = text("""
            SELECT * from dsm2 
            WHERE
                node in ('mid_gate_up','mid_gate_down','glc_gate_up','glc_gate_down','old_gate_up','old_gate_down');
            AND
                scenario_id in (SELECT id FROM scenarios WHERE name = :scenario)
            """)
        params = {"scenario": scenario}
    if backend:
        return pd.read_sql(q, engine, params=params, dtype_backend="pyarrow")

    return pd.read_sql(q, engine, params=params)
