import streamlit as st
from . import content
from . import collaborative
from . import hybrid

st.set_page_config(page_title="MovieAI Recommender", layout="centered")

st.title("🎬 MovieAI Recommender System")
st.markdown("Choose a recommendation method from the sidebar to get started.")

# Sidebar navigation
menu = ["Content-Based", "Collaborative", "Hybrid"]
choice = st.sidebar.radio("Select a Recommender:", menu)

if choice == "Content-Based":
    content.main()

elif choice == "Collaborative":
    collaborative.main()

elif choice == "Hybrid":
    hybrid.main()

