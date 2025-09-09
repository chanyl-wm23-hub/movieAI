# app.py
import streamlit as st
from recommender import content, collaborative, hybrid
import pandas as pd
from pathlib import Path

# -------------------------------
# Helper function to load CSV
# -------------------------------
@st.cache_data
def load_csv(filename):
    """Load a CSV file located next to app.py"""
    app_folder = Path(__file__).parent.resolve()  # folder where app.py resides
    csv_path = app_folder / filename
    if not csv_path.exists():
        st.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()  # empty dataframe if missing
    return pd.read_csv(csv_path)

# -------------------------------
# Load main CSV
# -------------------------------
df_main = load_csv("imdb_top_1000.csv")

# Stop execution if CSV not found
if df_main.empty:
    st.stop()

# -------------------------------
# Main App UI
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
