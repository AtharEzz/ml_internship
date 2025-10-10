import os
import streamlit as st
import pandas as pd
import pickle
from surprise import SVD

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# current_dir= os.path.dirname(os.path.abspath(__file__))
def load_model():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(current_dir, 'svd_model.pkl')
    model_path = os.path.join(CURRENT_DIR, 'svd_model.pkl')

    
    # with open('svd_model.pkl', 'rb') as f:
    with open(model_path, 'rb') as f:

        # return pickle.load(f)
        svd= pickle.load(f)

    return svd 


# def load_data():
    # train_ratings = pd.read_pickle('train_ratings.pkl')
    # movies = pd.read_pickle('movies.pkl')
    # return train_ratings, movies
    
def load_data():
    train_ratings_path = os.path.join(CURRENT_DIR, 'train_ratings.pkl')
    movies_path = os.path.join(CURRENT_DIR, 'movies.pkl')
    
    train_ratings = pd.read_pickle(train_ratings_path)
    movies = pd.read_pickle(movies_path)
    
    return train_ratings, movies

# Load model
svd = load_model()
train_ratings, movies = load_data()

# Get list of users
user_ids = sorted(train_ratings['user_id'].unique())

# App title
st.title("Movie Recommender System")
st.markdown("Built with **SVD Matrix Factorization**")

# User input
st.sidebar.header("Settings")
user_id = st.sidebar.selectbox("Select User ID", user_ids)
n_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

# Function to get recommendations
def get_recommendations(user_id, n_recs):
    # Get movies user has rated
    rated = set(train_ratings[train_ratings['user_id'] == user_id]['movie_id'])
    all_movies = set(movies['movie_id'])
    candidates = list(all_movies - rated)
    
    # Predict ratings
    predictions = []
    for mid in candidates:
        pred = svd.predict(uid=user_id, iid=mid)
        predictions.append((mid, pred.est))
    
    # Sort and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n_recs]

# Generate recommendations
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recs = get_recommendations(user_id, n_recs)
    
    st.subheader(f"Top {n_recs} Recommendations for User {user_id}")
    
    for i, (mid, score) in enumerate(recs, 1):
        title = movies[movies['movie_id'] == mid]['title'].iloc[0]
        st.write(f"{i}. **{title}** (Predicted Rating: {score:.2f})")
else:
    st.info("Select a user and click 'Get Recommendations' to start!")