import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class CollaborativeRecommender:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.user_similarity_matrix = None

    def preprocess_data(self):
        # Process the data to create a user-item matrix
        # Assuming 'id' is the user ID and you have an 'item_id' column for items
        user_item_matrix = self.data.pivot_table(index='id', columns='item_id', values='rating', fill_value=0)
        return user_item_matrix

    def calculate_similarity(self, user_item_matrix):
        # Calculate user-user similarity matrix
        user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
        self.user_similarity_matrix = cosine_similarity(user_item_matrix_sparse)

    def recommend_users(self, user_id, top_n=10):
        if self.user_similarity_matrix is None:
            raise Exception("User similarity matrix not calculated. Please run calculate_similarity first.")

        # Find the user index from user_id
        user_index = list(self.data['id']).index(user_id)

        # Get similarity scores for the user and sort them
        user_similarities = self.user_similarity_matrix[user_index]
        similar_users = sorted(list(enumerate(user_similarities)), key=lambda x: x[1], reverse=True)

        # Get top N similar users (excluding the user itself)
        top_users = [user for user in similar_users[1:top_n + 1]]
        return top_users
