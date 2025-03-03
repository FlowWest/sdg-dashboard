import folium
import streamlit as st
from streamlit_folium import st_folium

month_names = {
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
}


st.title("Map")

col_left, col_right = st.columns([1, 5])

with col_left:
    st.write("controls")
    st.selectbox("Select Year", options=range(2016, 2024))
    st.selectbox("Select Month", options=month_names)

with col_right:
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
    folium.Marker(
        [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
    ).add_to(m)

    st_data = st_folium(m, use_container_width=True)
