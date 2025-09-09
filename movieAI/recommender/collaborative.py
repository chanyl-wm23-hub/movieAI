# collaborative.py
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------
# Load CSVs
# -------------------------------
@st.cache_data
def load_ratings():
    csv_path = Path(__file__).parent.parent / "user_movie_rating.csv"
    if not csv_path.exists():
        st.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

@st.cache_data
def load_movies():
    csv_path = Path(__file__).parent.parent / "imdb_top_1000.csv"
    if not csv_path.exists():
        st.error(f"Movie CSV file not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

# -------------------------------
# Item-based Collaborative Filtering
# -------------------------------
@st.cache_data
def build_item_similarity(ratings_df):
    """
    Build item-item similarity matrix using user ratings
    """
    pivot = ratings_df.pivot_table(index='MovieID', columns='UserID', values='Rating').fillna(0)
    sim_matrix = cosine_similarity(pivot)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)
    return sim_df

def recommend_movies(selected_movie_id, movies_df, ratings_df, n_results=5, selected_genre=None):
    if selected_movie_id not in ratings_df['MovieID'].values:
        # fallback: return top-rated movies
        recs = movies_df.copy()
        if selected_genre:
            recs = recs[recs['Genre'] == selected_genre]
        return recs.sort_values('Rating', ascending=False).head(n_results)
    
    # Build similarity
    sim_df = build_item_similarity(ratings_df)
    
    # Find top similar movies
    sim_scores = sim_df[selected_movie_id].sort_values(ascending=False)
    sim_scores = sim_scores.drop(selected_movie_id, errors='ignore')
    top_ids = sim_scores.head(n_results).index.tolist()
    
    recommendations = movies_df[movies_df['ID'].isin(top_ids)]
    
    # Filter by genre if selected
    if selected_genre:
        recommendations = recommendations[recommendations['Genre'] == selected_genre]
    
    return recommendations.head(n_results)

# -------------------------------
# For testing in Streamlit
# -------------------------------
if __name__ == "__main__":
    st.title("Collaborative Filtering Test")

    movies_df = load_movies()
    ratings_df = load_ratings()
    
    if movies_df.empty or ratings_df.empty:
        st.stop()
    
    movie_list = [""] + movies_df['Title'].tolist()
    selected_movie_title = st.selectbox("Select a movie:", options=movie_list)
    
    selected_movie_id = None
    if selected_movie_title:
        row = movies_df[movies_df['Title'] == selected_movie_title]
        if not row.empty:
            selected_movie_id = row.iloc[0]['ID']
    
    selected_genre = st.selectbox("Select a genre (optional):", options=[""] + movies_df['Genre'].dropna().unique().tolist())
    n_results = st.slider("Number of recommendations:", min_value=5, max_value=15, value=5)
    generate = st.button("Generate Recommendations")
    
    if generate:
        recommendations = recommend_movies(selected_movie_id, movies_df, ratings_df, n_results, selected_genre)
        if recommendations.empty:
            st.write("No recommendations found.")
        else:
            st.table(recommendations[['Title','Genre','Rating','Year']])
