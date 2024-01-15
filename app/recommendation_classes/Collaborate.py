import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def create_user_item_matrix(df, user_col, item_col, rating_col):
    """Create a user-item matrix."""
    user_item_matrix = df.pivot(index=user_col, columns=item_col, values=rating_col)
    user_item_matrix.fillna(0, inplace=True)
    return user_item_matrix

def feature_similarity(df, user_ids, features):
    """Calculate user similarity based on features."""
    feature_matrix = df.set_index('id')[features]
    feature_distances = distance.cdist(feature_matrix, feature_matrix, 'euclidean')
    feature_similarity = 1 / (1 + feature_distances)  # Convert distances to similarity
    return feature_similarity

def collaborative_filtering(user_item_matrix):
    """Calculate user similarity using Collaborative Filtering."""
    user_similarity = cosine_similarity(user_item_matrix)
    return user_similarity

def combined_similarity(df, user_ids, features, user_item_matrix):
    """Combine feature-based and collaborative filtering similarities."""
    feature_sim = feature_similarity(df, user_ids, features)
    collab_sim = collaborative_filtering(user_item_matrix)
    combined_sim = (feature_sim + collab_sim) / 2  # Simple average, can be weighted
    return combined_sim

# Example usage:
file_path = 'user_data.csv'  # Update with your file path
df = load_data(file_path)

user_ids = df['id'].unique()
features = ['age', 'location', 'interests']  # Update with actual feature columns
user_item_matrix = create_user_item_matrix(df, 'user_id', 'item_id', 'rating')

# Calculate combined similarity matrix
combined_similarity_matrix = combined_similarity(df, user_ids, features, user_item_matrix)

# Further steps to make predictions or recommendations would go here
