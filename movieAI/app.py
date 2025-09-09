# app.py
import streamlit as st
from recommender import content, collaborative, hybrid
import pandas as pd
import os
from pathlib import Path

# -------------------------------
# Helper function to load CSVs
# -------------------------------
@st.cache_data
def load_csv(filename):
    """Load a CSV file located next to app.py"""
    app_folder = Path(os.getcwd())  # assumes Streamlit launched from movieAI folder
    csv_path = app_folder / filename
    if not csv_path.exists():
        st.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()  # return empty dataframe if file missing
    return pd.read_csv(csv_path)

# Load data
df_main = load_csv("imdb_top_1000.csv")
# df_additional = load_csv("imdb_top_1000_additional.csv")  # uncomment if you have a second CSV

# -------------------------------
# Main App
# -------------------------------
st.title("Movie Recommender App 🎬")

st.write("Welcome to the Movie AI Recommender!")

# Tabs for different recommendation types
tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

with tab1:
    st.header("Content-Based Recommendations")
    content.main(df_main)  # pass dataframe to content module

with tab2:
    st.header("Collaborative Filtering")
    collaborative.main(df_main)

with tab3:
    st.header("Hybrid Recommendations")
    hybrid.main(df_main)
