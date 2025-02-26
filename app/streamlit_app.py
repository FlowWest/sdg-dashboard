import streamlit as st

st.set_page_config(layout="wide")

pg = st.navigation(
    {
        "Home": [st.Page("home.py", title="Home")],
        "Historically Based Scenario": [
            st.Page("page_1.py", title="Single Year Historically Based Simulation"),
            st.Page("page_2.py", title="Scenario Comparison"),
            # st.Page("page_3.py", title="20-25 Year Historically Based Simulation"),
        ],
        "CalSim Based Scenario": [
            st.Page("page_4.py", title="100 Year CalSim Based Simulation")
        ],
        "Data Explorer": [st.Page("data_explorer.py", title="Data Explorer")],
        "Scenario Inventory": [
            st.Page("scenario_inventory.py", title="Scenario Inventory")
        ],
    }
)
pg.run()
