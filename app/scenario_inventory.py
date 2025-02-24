import streamlit as st
import pandas as pd

from db import engine, get_all_scenarios

unique_scenarios_query = (
    'select name as "Scenario", comments as "Comments" from scenarios;'
)

st.title("Scenarios Inventory")
st.write("Collection of scenarios available for analysis in dashboard")

df = get_all_scenarios()
st.dataframe(df, hide_index=True)


st.markdown(
    """
    to add additional scenarios to the SDG Dashboard use the accompanying `sdgtools` python package. More info available
    at the github [repository](https://github.com/FlowWest/sdgtools)
    """
)
