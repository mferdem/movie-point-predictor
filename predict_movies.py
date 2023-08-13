import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load the prediction data
prediction_data = pd.read_csv('sample_prediction.csv')

# Prepare prediction features
prediction_features = prediction_data[['imdb_point', 'release_year', 'runtime', 'num_votes', 'is_action', 'is_adventure', 'is_animation', 'is_biography', 'is_comedy', 'is_crime', 'is_drama', 'is_family', 'is_fantasy', 'is_filmnoir', 'is_history', 'is_horror', 'is_music', 'is_musical', 'is_mystery', 'is_romance', 'is_scifi', 'is_sport', 'is_thriller', 'is_war', 'is_western']].values

# Normalize the features
scaler = StandardScaler()
prediction_features = scaler.fit_transform(prediction_features)

# Load the trained model
model = keras.models.load_model('movie_point_predictor_model.keras')

# Make predictions
predictions = model.predict(prediction_features)

# Print the predicted ratings (capped at 10)
for i, prediction in enumerate(predictions):
    predicted_rating = min(prediction[0], 10.0)  # Cap the predicted rating at 10
    print(f"Movie {i+1}: Predicted Rating = {predicted_rating:.2f}")
