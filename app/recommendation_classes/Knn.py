import pandas as pd
import numpy as np
from scipy.spatial import distance
import operator

from app.model.User import User


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def user_similarity_matrix(user_ids, df, features):
    """Calculate user similarity matrix."""
    similarity_matrix = np.zeros((len(user_ids), len(user_ids)))

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user1 = df[df['id'] == user_ids[i]]
            user2 = df[df['id'] == user_ids[j]]

            distances = [distance.euclidean(user1[feature].values, user2[feature].values) for feature in features]
            similarity_matrix[i, j] = similarity_matrix[j, i] = sum(distances)

    return similarity_matrix

def predict_score(user_id, df, similarity_matrix, K=100):
    """Predict scores and recommend users."""
    user_data = df[df['id'] == user_id]
    if user_data.empty:
        print(f"User with ID {user_id} not found.")
        return

    print('Selected User: ', user_data['username'].values[0])

    def get_neighbors(base_user, K):
        distances = [(index, similarity_matrix[user_data.index[0], index]) for index in df.index if index != user_data.index[0]]
        distances.sort(key=operator.itemgetter(1), reverse=True)
        return distances[:K]

    neighbors = get_neighbors(user_data, K)
    neighbor_data_list = []

    #print('\nRecommended Users: \n')
    for neighbor in neighbors:
        neighbor_id, similarity = neighbor
        # Create an independent copy of the DataFrame slice
        neighbor_data = df.loc[neighbor_id].copy()

        # Now safely add the similarity value
        neighbor_data['similarity'] = similarity

        # Convert to dictionary and then to a Pydantic model
        user_data = User(**neighbor_data.to_dict())
        neighbor_data_list.append(user_data)

    return neighbor_data_list
    #print('\n')
    #avg_rating /= K
    #print(f'The predicted average rating for {user_data["username"].values[0]} is: {avg_rating:.4f}')
    #print(f'The actual average rating for {user_data["username"].values[0]} is: {user_data["profile_score"].values[0]}')
