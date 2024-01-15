import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.model.User import User

class SVMRecommender:
    def __init__(self, file_path, features, target):
        self.data = pd.read_csv(file_path)
        self.features = features
        self.target = target
        self.model = None

    def preprocess_data(self):
        # Handling categorical data
        label_encoder = LabelEncoder()
        for feature in self.features:
            if self.data[feature].dtype == 'object':
                self.data[feature] = label_encoder.fit_transform(self.data[feature])
                # For non-numeric columns, fill NaNs with a placeholder or the most frequent value
                self.data[feature].fillna(self.data[feature].mode()[0], inplace=True)
            else:
                # For numeric columns, fill NaNs with the mean
                self.data[feature].fillna(self.data[feature].mean(), inplace=True)

        # Extract features and target variable
        X = self.data[self.features]
        y = self.data[self.target]

        # Normalizing the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y

    def train(self):
        X, y = self.preprocess_data()

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the model
        self.model = SVC(kernel='linear')  # You can adjust the kernel and other parameters as needed
        self.model.fit(X_train, y_train)  # Make sure X_train and y_train are defined here

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")

    def predict(self, features):
        if self.model is None:
            raise Exception("Model not trained. Please train the model before prediction.")

        # Prediction
        return self.model.predict([features])

    def recommend_users(self, user_id):
        if self.model is None:
            raise Exception("Model not trained. Please train the model before prediction.")

        # Filter out the current user
        other_users = self.data[self.data['id'] != user_id]

        other_users_features = other_users[self.features].values
        predictions = self.model.predict(other_users_features)
        print("Raw predictions:", predictions)  # Check these values

        other_users['similarity'] = predictions

        # Sort by predicted score
        recommended_users = other_users.sort_values(by='similarity', ascending=False)

        print(recommended_users)
        # Convert to User instances
        recommended_users_list = [User(**user.to_dict()) for index, user in recommended_users.iterrows()]


        return recommended_users_list