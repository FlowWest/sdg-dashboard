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
