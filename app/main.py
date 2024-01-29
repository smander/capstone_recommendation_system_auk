import random
from http.client import HTTPException

from fastapi import FastAPI, HTTPException, Query

from app.recommendation_classes.Collaborate import CollaborativeRecommender
from app.recommendation_classes.Knn import *
from app.recommendation_classes.SVM import *
from app.requests.recommendations.post import RecommendationParams
from app.db import database, User
from app.formula.CCR import calculate_CCR


app = FastAPI(title="Capstone Project")

@app.get("/")
async def read_root():
    default_response = {
        'code': 2010,
    }
    return default_response

@app.post("/recommendation")
async def get_recommendations(params: RecommendationParams):
    file_path = 'test_data/convertcsv (1).csv'
    df = load_data(file_path)

    if params.model == "knn":
        user_ids = df['id'].unique()
        features = ['profile_score', 'age']  # Adjust features as needed
        similarity_matrix_custom = user_similarity_matrix(user_ids, df, features)

        score = predict_score(30, df, similarity_matrix_custom)
    elif params.model == "svm":
        feature_columns = ['profile_score', 'age']  # Adjust as needed
        target_column = 'profile_score'  # Adjust as needed

        svm_recommender = SVMRecommender(file_path, feature_columns, target_column)
        svm_recommender.train()

        score = svm_recommender.recommend_users(params.user_id)
    elif params.model == "cff":
        # Usage
        collab_recommender = CollaborativeRecommender(file_path)
        user_item_matrix = collab_recommender.preprocess_data()
        collab_recommender.calculate_similarity(user_item_matrix)

        # Get recommendations for a specific user
        score = collab_recommender.recommend_users(params.user_id, top_n=50)

    return score


@app.post("/ccr")
async def calculate_ccr_example():
    # Number of users
    num_users = 100

    # Simulating recommendations (for simplicity, each user has 5 recommendations)
    recommendations = {user: random.sample(range(num_users), 5) for user in range(num_users)}

    # Simulating similarity scores between users (random values between 0 and 1)
    similarity_scores = {(i, j): random.random() for i in range(num_users) for j in range(num_users)}

    # Simulating completed connections (randomly deciding if a connection is completed)
    completed_connections = {(i, j): random.choice([0, 1]) for i in range(num_users) for j in range(num_users)}

    CCR = calculate_CCR(recommendations, similarity_scores, completed_connections)
    return CCR

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()
    # create a dummy entry
    await User.objects.get_or_create(email="test@test.com")


@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()
