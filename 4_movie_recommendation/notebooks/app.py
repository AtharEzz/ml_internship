import os
import streamlit as st
import pandas as pd
import pickle
from surprise import SVD

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
def load_model():
    
    model_path = os.path.join(CURRENT_DIR, 'svd_model.pkl')

    
    with open(model_path, 'rb') as f:

        svd= pickle.load(f)

    return svd 



    
# def load_data():
    # train_ratings_path = os.path.join(CURRENT_DIR, 'train_ratings.pkl')
    # movies_path = os.path.join(CURRENT_DIR, 'movies.pkl')
    
    # train_ratings = pd.read_pickle(train_ratings_path)
    # movies = pd.read_pickle(movies_path)
    
    # return train_ratings, movies
    
    
def load_user_based_components():
    similarity_path = os.path.join(CURRENT_DIR, 'user_similarity.pkl')
    train_path = os.path.join(CURRENT_DIR, 'train_ratings.pkl')
    movies_path = os.path.join(CURRENT_DIR, 'movies.pkl')
    
    user_similarity = pd.read_pickle(similarity_path)
    train_ratings = pd.read_pickle(train_path)
    movies = pd.read_pickle(movies_path)
    
    return user_similarity, train_ratings, movies

# Load model
svd = load_model()
user_similarity_df, train_ratings, movies = load_user_based_components()

# train_ratings, movies = load_data()

# # Get list of users
# user_ids = sorted(train_ratings['user_id'].unique())

# # App title
# st.title("Movie Recommender System")
# st.markdown("Built with **SVD Matrix Factorization**")

# # User input
# st.sidebar.header("Settings")
# user_id = st.sidebar.selectbox("Select User ID", user_ids)
# n_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

# # Function to get recommendations
# def get_recommendations(user_id, n_recs):
    # # Get movies user has rated
    # rated = set(train_ratings[train_ratings['user_id'] == user_id]['movie_id'])
    # all_movies = set(movies['movie_id'])
    # candidates = list(all_movies - rated)
    
    # # Predict ratings
    # predictions = []
    # for mid in candidates:
        # pred = svd.predict(uid=user_id, iid=mid)
        # predictions.append((mid, pred.est))
    
    # # Sort and return top N
    # predictions.sort(key=lambda x: x[1], reverse=True)
    # return predictions[:n_recs]

# # Generate recommendations
# if st.sidebar.button("Get Recommendations"):
    # with st.spinner("Generating recommendations..."):
        # recs = get_recommendations(user_id, n_recs)
    
    # st.subheader(f"Top {n_recs} Recommendations for User {user_id}")
    
    # for i, (mid, score) in enumerate(recs, 1):
        # title = movies[movies['movie_id'] == mid]['title'].iloc[0]
        # st.write(f"{i}. **{title}** (Predicted Rating: {score:.2f})")
# else:
    # st.info("Select a user and click 'Get Recommendations' to start!")
    
    
# Get list of users (from train_ratings)
user_ids = sorted(train_ratings['user_id'].unique())

# App title
st.title("Movie Recommender System")
st.markdown("Compare **SVD Matrix Factorization** vs **User-Based Collaborative Filtering**")

# Sidebar controls
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Recommendation Model",
    ["SVD (Precision@5: 0.130)", "User-Based CF (Precision@5: 0.120)"]
)
user_id = st.sidebar.selectbox("Select User ID", user_ids)
n_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

# SVD Recommendation Function
def get_svd_recommendations(user_id, n_recs):
    rated = set(train_ratings[train_ratings['user_id'] == user_id]['movie_id'])
    all_movies = set(movies['movie_id'])
    candidates = list(all_movies - rated)
    
    predictions = []
    for mid in candidates:
        pred = svd.predict(uid=user_id, iid=mid)
        predictions.append((mid, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n_recs]

# User-Based CF Recommendation Function
def get_user_based_recommendations(user_id, n_recs):
    if user_id not in user_similarity_df.index:
        return []
    
    # Get top 10 similar users
    top_similar = user_similarity_df.loc[user_id].sort_values(ascending=False)[1:11]
    top_similar = top_similar[top_similar > 0.1]
    
    if len(top_similar) < 2:
        return []
    
    # Get ratings from similar users
    similar_ratings = train_ratings[train_ratings['user_id'].isin(top_similar.index)].copy()
    similar_ratings['similarity'] = similar_ratings['user_id'].map(top_similar)
    similar_ratings['weighted_ratings'] = similar_ratings['rating'] * similar_ratings['similarity']
    
    # Aggregate by movie
    grouped = similar_ratings.groupby('movie_id').agg({
        'similarity': 'sum',
        'weighted_ratings': 'sum',
        'user_id': 'count'
    })
    grouped.rename(columns={'user_id': 'similar_count'}, inplace=True)
    grouped = grouped[grouped['similar_count'] >= 2]
    grouped['pred_score'] = grouped['weighted_ratings'] / grouped['similarity']
    
    # Add popularity tie-breaker
    movie_popularity = train_ratings['movie_id'].value_counts()
    grouped['popularity'] = grouped.index.map(movie_popularity).fillna(0)
    grouped['final_score'] = grouped['pred_score'] + 0.001 * grouped['popularity']
    
    # Filter out already watched movies
    watched = set(train_ratings[train_ratings['user_id'] == user_id]['movie_id'])
    recommendations = grouped[~grouped.index.isin(watched)]
    
    if recommendations.empty:
        return []
    
    top_recs = recommendations['final_score'].sort_values(ascending=False).head(n_recs)
    return [(mid, score) for mid, score in top_recs.items()]

# Generate recommendations
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        if "SVD" in model_choice:
            recs = get_svd_recommendations(user_id, n_recs)
            model_name = "SVD Matrix Factorization"
        else:
            recs = get_user_based_recommendations(user_id, n_recs)
            model_name = "User-Based Collaborative Filtering"
    
    if recs:
        st.subheader(f"Top {n_recs} {model_name} Recommendations for User {user_id}")
        for i, (mid, score) in enumerate(recs, 1):
            title = movies[movies['movie_id'] == mid]['title'].iloc[0]
            if "SVD" in model_choice:
                st.write(f"{i}. **{title}** (Predicted Rating: {score:.2f})")
            else:
                st.write(f"{i}. **{title}** (Predicted Score: {score:.2f})")
    else:
        st.warning("No recommendations available for this user with the selected model.")
else:
    st.info("👈 Select a model, user, and click 'Get Recommendations' to start!")