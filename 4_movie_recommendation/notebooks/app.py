# app.py
import streamlit as st
import pandas as pd
import pickle
from surprise import SVD

# Load data and model

def load_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)


def load_data():
    train_ratings = pd.read_pickle('train_ratings.pkl')
    movies = pd.read_pickle('movies.pkl')
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