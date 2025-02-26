import streamlit as st
import pandas as pd
import altair as alt
import pickle
from figures_functions import *
#import plotly.express as px
import datetime
import numpy as np
from db import (
    get_scenario_year_data,
    get_all_scenarios,
    generate_scenario_year_data,
    get_filter_nodes_for_gate,
    get_available_years
)

@st.cache_data
def load_scenario_data(scenario, year):
    """Cache the scenario data loading to prevent unnecessary database queries"""
    scenario_year_data = get_scenario_year_data(scenario, year)
    scenario_data = generate_scenario_year_data(scenario_year_data)
    return scenario_year_data, scenario_data

def get_scenario_and_year_selection(column, scenario_key, year_key, previous_scenario_key, available_years_key, previous_year_key):
    """Function to handle the scenario selection and dynamic year loading."""
    
    # Scenario selection
    selected_model = st.selectbox(f"Select Scenario:", scenario_list, key=scenario_key)

    # Load years dynamically based on selected scenario
    if selected_model != st.session_state[previous_scenario_key]:
        st.session_state[available_years_key] = get_available_years(selected_model)
        st.session_state[previous_scenario_key] = selected_model
        st.session_state[previous_year_key] = None  # Reset year when scenario changes

    # Year selection
    if st.session_state[available_years_key]:
        selected_year = st.selectbox(f"Select Year:", st.session_state[available_years_key], key=year_key)
        # st.write(f"Selected Scenario: {selected_model}, Year: {selected_year}")
    else:
        selected_year = None

    return selected_model, selected_year

# Initialize Streamlit app
st.title("Exploratory Multi-Year Data Visualizations")

if "previous_year_1" not in st.session_state:
    st.session_state.previous_year_1 = None
if "previous_scenario_1" not in st.session_state:
    st.session_state.previous_scenario_1 = None
if "scenario_data_1" not in st.session_state:
    st.session_state.scenario_data_1 = None
    st.session_state.scenario_year_data_1 = None
if "available_years_1" not in st.session_state:
    st.session_state.available_years_1 = []

if "previous_year_2" not in st.session_state:
    st.session_state.previous_year_2 = None
if "previous_scenario_2" not in st.session_state:
    st.session_state.previous_scenario_2 = None
if "scenario_data_2" not in st.session_state:
    st.session_state.scenario_data_2 = None
    st.session_state.scenario_year_data_2 = None
if "available_years_2" not in st.session_state:
    st.session_state.available_years_2 = []

# Step 1: Load available scenarios first
scenarios = get_all_scenarios()
scenario_list = scenarios["Scenario"].tolist()

# Step 2: Create two columns
col1, col2 = st.columns(2)

# Step 3: Call function for both columns
with col1:
    selected_model_1, selected_year_1 = get_scenario_and_year_selection(
        column="1", 
        scenario_key="scenario_select_1", 
        year_key="year_select_1", 
        previous_scenario_key="previous_scenario_1", 
        available_years_key="available_years_1", 
        previous_year_key="previous_year_1"
    )

with col2:
    selected_model_2, selected_year_2 = get_scenario_and_year_selection(
        column="2", 
        scenario_key="scenario_select_2", 
        year_key="year_select_2", 
        previous_scenario_key="previous_scenario_2", 
        available_years_key="available_years_2", 
        previous_year_key="previous_year_2"
    )

submit_button = st.button("Submit")

if submit_button:
    # Fetch data for the selected scenarios and years
    scenario_year_data_1, data_1 = load_scenario_data(selected_model_1, selected_year_1)
    scenario_year_data_2, data_2 = load_scenario_data(selected_model_2, selected_year_2)
    
    # Display the data previews
    with col1:
        st.success(f"Scenario 1 Loaded: {selected_model_1} ({selected_year_1})")
        st.write(data_1["flow"], use_container_width=True)  
    with col2:
        st.success(f"Scenario 2 Loaded: {selected_model_2} ({selected_year_2})")
        st.dataframe(data_2["flow"], use_container_width=True)
else:
    st.write("Please select the scenarios and years, then click 'Submit' to preview the data.")