import streamlit as st
import pandas as pd
from recommender import content, collaborative

def hybrid_recommendations(movie_title, movies_df, ratings_df, top_n=10, alpha=0.5):
    """
    Hybrid recommender combining content-based and collaborative filtering.
    alpha = weight for content-based (0 to 1)
    (1-alpha) = weight for collaborative
    """
    # Get content-based results
    content_recs = content.get_content_recommendations(movie_title, movies_df, top_n)
    content_recs["content_score"] = [1.0 - (i/top_n) for i in range(len(content_recs))]

    # Get collaborative results
    collab_recs = collaborative.get_collaborative_recommendations(movie_title, ratings_df, movies_df, top_n)
    collab_recs["collab_score"] = [1.0 - (i/top_n) for i in range(len(collab_recs))]

    # Merge both recommendations on title
    hybrid_df = pd.merge(content_recs, collab_recs, on="Series_Title", how="outer")

    # Fill NaNs with 0
    hybrid_df["content_score"] = hybrid_df["content_score"].fillna(0)
    hybrid_df["collab_score"] = hybrid_df["collab_score"].fillna(0)

    # Weighted final score
    hybrid_df["final_score"] = alpha * hybrid_df["content_score"] + (1 - alpha) * hybrid_df["collab_score"]

    # Sort by final score
    hybrid_df = hybrid_df.sort_values("final_score", ascending=False).head(top_n)

    return hybrid_df[["Series_Title", "final_score"]]


def main():
    st.subheader("🔀 Hybrid Filtering Recommender")

    # Load data
    movies_df = pd.read_csv("data/imdb_top_1000.csv")
    ratings_df = pd.read_csv("data/user_movie_rating.csv")

    # User input
    movie_list = movies_df["Series_Title"].dropna().unique()
    movie_choice = st.selectbox("Choose a movie:", movie_list)

    alpha = st.slider("Weight for Content-based (vs Collaborative)", 0.0, 1.0, 0.5, 0.1)
    top_n = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Get Hybrid Recommendations"):
        results = hybrid_recommendations(movie_choice, movies_df, ratings_df, top_n, alpha)
        st.write(results)


if __name__ == "__main__":
    main()
