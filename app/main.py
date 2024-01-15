import random
from http.client import HTTPException

from fastapi import FastAPI, HTTPException, Query

from app.recommendation_classes.Knn import *
from app.recommendation_classes.SVM import *
from app.requests.recommendations.post import RecommendationParams
from app.db import database, User


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

        user_id = 30  # Replace with the actual user ID
        score = predict_score(user_id, df, similarity_matrix_custom)
    elif params.model == "svm":
        feature_columns = ['profile_score', 'age']  # Adjust as needed
        target_column = 'profile_score'  # Adjust as needed

        svm_recommender = SVMRecommender(file_path, feature_columns, target_column)
        svm_recommender.train()

        score = svm_recommender.recommend_users(params.user_id)

    return score

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
