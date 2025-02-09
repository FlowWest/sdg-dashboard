import streamlit as st
import pandas as pd

from sqlalchemy import create_engine


@st.cache_resource
def get_db_connection():
    db_url = st.secrets["database"]["url"]
    return create_engine(db_url)


engine = get_db_connection()


def get_all_scenarios():
    q = """ SELECT name as "Scenario", comments as "Comments" from scenarios; """
    return pd.read_sql(q, engine)


def get_gateop_elevations(scenario, year=None):
    if year:
        q = f"""
            SELECT * from dsm2 where node in ('mid_gate_up','mid_gate_down','glc_gate_up','glc_gate_down','old_gate_up','old_gate_down') and date_part('year', datetime) = {year} and where scenario_id in (select id from scenarios where name = '{scenario}');
            """
    else:
        q = """
            SELECT * from dsm2 where node in ('mid_gate_up','mid_gate_down','glc_gate_up','glc_gate_down','old_gate_up','old_gate_down');
            """
    return pd.read_sql(q, engine)
