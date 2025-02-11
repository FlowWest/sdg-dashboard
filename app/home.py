import streamlit as st


# Main content
st.title("Delta Simulation Tool 🌊")

st.markdown(
    """
    <h3 style='text-align: center; color: #666666;'>
    Water Resource Planning & Analysis Platform
    </h3>
""",
    unsafe_allow_html=True,
)

# Introduction section
st.markdown("""
This tool provides multiple simulation scenarios for water resource planning in the Delta region:
""")

# Create three columns for the main scenario types
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Historically Based")
    st.write("""
    Run simulations based on historical data with varying timeframes:
    - Single year analysis
    - 3-5 year medium-term analysis
    - 20-25 year long-term analysis
    """)

with col2:
    st.subheader("CalSim Based")
    st.write("""
    Explore extended simulations using CalSim modeling:
    - 100-year scenario analysis
    - Climate change considerations
    - System-wide impacts
    """)

with col3:
    st.subheader("Analysis Tools")
    st.write("""
    Additional resources:
    - Data Explorer for detailed analysis
    - Scenario Inventory tracking
    - Comparative visualization tools
    """)

# Quick start guide
st.markdown("---")
st.subheader("Quick Start Guide")
st.write("""
1. Choose a scenario type from the navigation menu on the left
2. Select your desired simulation timeframe
3. Configure parameters and run your simulation
4. Explore results in the Data Explorer
5. Save and manage scenarios in the Scenario Inventory
""")

# Additional information or disclaimers if needed
st.markdown("---")
st.caption("""
This simulation tool is designed to support water resource planning and decision-making. 
Results should be interpreted within the context of your specific planning needs.
""")
