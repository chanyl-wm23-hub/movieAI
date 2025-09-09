import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("imdb_top_1000.csv")

    # Handle missing values
    df.fillna('', inplace=True)

    # Make sure Genre is clean
    df['Genre'] = df['Genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    
    return df

# -----------------------
# Build KNN Model
# -----------------------
@st.cache_resource
def build_knn_model(df):
    # Use Genre, Director, Star1, Star2, Star3, Description
    df['combined_features'] = (
        df['Genre'].apply(lambda x: " ".join(x)) + " " +
        df['Director'] + " " +
        df['Star1'] + " " +
        df['Star2'] + " " +
        df['Star3'] + " " +
        df['Overview']
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    feature_matrix = vectorizer.fit_transform(df['combined_features'])

    # KNN with cosine similarity
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(feature_matrix)

    return knn, feature_matrix

# -----------------------
# Recommendation Function
# -----------------------
def recommend_movies(movie_title, df, knn, feature_matrix, top_n=10):
    try:
        movie_idx = df[df['Series_Title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return None
    
    distances, indices = knn.kneighbors(feature_matrix[movie_idx], n_neighbors=top_n+1)
    
    recommendations = df.iloc[indices[0][1:]][['Series_Title', 'Genre', 'IMDB_Rating']]
    recommendations['similarity_score'] = 1 - distances[0][1:]
    
    return recommendations

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.title("🎬 Content-Based Movie Recommender (KNN)")

    df = load_data()
    knn, feature_matrix = build_knn_model(df)

    movie_list = df['Series_Title'].sort_values().tolist()
    selected_movie = st.selectbox("Choose a movie you like:", movie_list)

    if st.button("Recommend"):
        recommendations = recommend_movies(selected_movie, df, knn, feature_matrix, top_n=10)
        if recommendations is not None:
            st.subheader(f"Movies similar to: {selected_movie}")
            st.dataframe(recommendations)
        else:
            st.error("Movie not found in dataset.")

if __name__ == "__main__":
    main()
