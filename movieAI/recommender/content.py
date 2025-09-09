# content_based.py
import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------------------
# Load CSV
# -------------------------------
@st.cache_data
def load_data():
    """Load IMDb CSV file located next to this script"""
    csv_path = Path(__file__).parent / "imdb_top_1000.csv"
    if not csv_path.exists():
        st.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

# -------------------------------
# Simple content-based recommendation logic
# -------------------------------
def recommend_movies(df, selected_movie, selected_genre, n_results):
    """
    Placeholder content-based logic:
    - Filter by genre if selected
    - Exclude the selected movie
    - Return top movies sorted by rating
    """
    recs = df.copy()

    if selected_genre:
        recs = recs[recs['Genre'] == selected_genre]

    if selected_movie:
        recs = recs[recs['Title'] != selected_movie]

    return recs.sort_values('Rating', ascending=False).head(n_results)

# -------------------------------
# Main UI
# -------------------------------
def main():
    st.title("Content-Based Movie Recommender 🎬")

    df = load_data()
    if df.empty:
        return

    # Split screen
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Recommendation Settings")

        # Movie selection (optional)
        movie_list = [""] + df['Title'].tolist()
        selected_movie = st.selectbox("Select a movie (optional):", options=movie_list)

        # Genre selection (single select)
        genres = [""] + df['Genre'].dropna().unique().tolist()
        selected_genre = st.selectbox("Select a genre (optional):", options=genres)

        # Show movie details
        if selected_movie:
            movie_details = df[df['Title'] == selected_movie].iloc[0]
            st.markdown("**Movie Details:**")
            st.write(f"**Title:** {movie_details['Title']}")
            st.write(f"**ID:** {movie_details.get('ID', 'N/A')}")
            st.write(f"**Genre:** {movie_details['Genre']}")
            st.write(f"**Rating:** {movie_details['Rating']}")
            st.write(f"**Year Released:** {movie_details.get('Year', 'N/A')}")

        # Number of recommendations
        n_results = st.slider("Number of recommendations:", min_value=5, max_value=15, value=5)

        # Generate button
        generate = st.button("Generate Recommendations")

    with right_col:
        st.subheader("Recommended Movies")
        if generate:
            recommendations = recommend_movies(df, selected_movie, selected_genre, n_results)
            if recommendations.empty:
                st.write("No recommendations found.")
            else:
                st.table(recommendations[['Title', 'Genre', 'Rating', 'Year']])

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    main()
