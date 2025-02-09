import streamlit as st
from sqlalchemy import create_engine


@st.cache_resource
def get_db_connection():
    db_url = st.secrets["database"]["url"]
    return create_engine(db_url)


engine = get_db_connection()
