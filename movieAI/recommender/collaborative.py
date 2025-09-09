# app_collaborative.py

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    ratings = pd.read_csv("data/user_movie_rating.csv")
    movies = pd.read_csv("data/imdb_top_1000.csv")
    return ratings, movies

# Build user-item matrix
def build_user_item_matrix(ratings):
    return ratings.pivot_table(
        index="User_ID", 
        columns="Movie_ID", 
        values="Rating"
    ).fillna(0)

# Collaborative filtering using cosine similarity
def collaborative_filtering(movie_id, user_item_matrix, movies, top_n=10):
    # Compute cosine similarity between movies
    movie_similarity = cosine_similarity(user_item_matrix.T)
    similarity_df = pd.DataFrame(
        movie_similarity, 
        index=user_item_matrix.columns, 
        columns=user_item_matrix.columns
    )
    
    # Get similar movies
    if movie_id not in similarity_df.index:
        return pd.DataFrame(columns=["Series_Title", "IMDB_Rating"])
    
    similar_scores = similarity_df[movie_id].sort_values(ascending=False)[1:top_n+1]
    
    # Join with movie details
    result = movies[movies["Movie_ID"].isin(similar_scores.index)]
    result = result[["Series_Title", "IMDB_Rating", "Genre", "Director"]]
    return result

# Streamlit UI
def main():
    st.title("🎬 Collaborative Filtering Recommender (Cosine Similarity)")
    
    ratings, movies = load_data()
    user_item_matrix = build_user_item_matrix(ratings)
    
    st.sidebar.header("Select a Movie")
    movie_list = movies["Series_Title"].tolist()
    selected_movie = st.sidebar.selectbox("Pick a movie:", movie_list)
    
    if st.sidebar.button("Recommend"):
        # Get movie ID
        movie_id = movies[movies["Series_Title"] == selected_movie]["Movie_ID"].values[0]
        
        recommendations = collaborative_filtering(movie_id, user_item_matrix, movies)
        
        st.subheader(f"Movies similar to: {selected_movie}")
        if recommendations.empty:
            st.warning("No recommendations found for this movie.")
        else:
            st.dataframe(recommendations)

if __name__ == "__main__":
    main()

