# hybrid.py
import pandas as pd
import streamlit as st
from pathlib import Path
from collaborative import recommend_movies as collab_recommend
from content_based import recommend_movies as content_recommend

# -------------------------------
# Load CSVs
# -------------------------------
@st.cache_data
def load_movies():
    csv_path = Path(__file__).parent.parent / "imdb_top_1000.csv"
    if not csv_path.exists():
        st.error(f"Movie CSV file not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

@st.cache_data
def load_ratings():
    csv_path = Path(__file__).parent.parent / "user_movie_rating.csv"
    if not csv_path.exists():
        st.error(f"User rating CSV not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

# -------------------------------
# Hybrid Recommendation Logic
# -------------------------------
def recommend_movies(selected_movie_id, selected_movie_title, movies_df, ratings_df, n_results=5, selected_genre=None, weight_content=0.5, weight_collab=0.5):
    """
    Hybrid recommendation:
    - Get content-based and collaborative recommendations separately
    - Assign weights to each method and combine
    """
    # 1️⃣ Content-Based Recommendations
    content_recs = content_recommend(movies_df, selected_movie_title, selected_genre, n_results*2)
    content_recs = content_recs.copy()
    content_recs['score_content'] = range(len(content_recs), 0, -1)  # higher rank = higher score

    # 2️⃣ Collaborative Recommendations
    collab_recs = collab_recommend(selected_movie_id, movies_df, ratings_df, n_results*2, selected_genre)
    collab_recs = collab_recs.copy()
    collab_recs['score_collab'] = range(len(collab_recs), 0, -1)  # higher rank = higher score

    # 3️⃣ Merge scores
    merged = pd.merge(content_recs, collab_recs, on='ID', how='outer', suffixes=('_content','_collab')).fillna(0)
    merged['hybrid_score'] = merged['score_content']*weight_content + merged['score_collab']*weight_collab

    # 4️⃣ Sort by hybrid score
    recommendations = merged.sort_values('hybrid_score', ascending=False).head(n_results)

    return recommendations[['Title','Genre','Rating','Year']]

# -------------------------------
# For testing in Streamlit
# -------------------------------
if __name__ == "__main__":
    st.title("Hybrid Filtering Test")

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
        recommendations = recommend_movies(selected_movie_id, selected_movie_title, movies_df, ratings_df, n_results, selected_genre)
        if recommendations.empty:
            st.write("No recommendations found.")
        else:
            st.table(recommendations)
