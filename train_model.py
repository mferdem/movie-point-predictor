import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Load the training data
data = pd.read_csv('sample_data.csv')

# Prepare features and labels
train_features = data[['imdb_point', 'release_year', 'runtime', 'num_votes', 'is_action', 'is_adventure', 'is_animation', 'is_biography', 'is_comedy', 'is_crime', 'is_drama', 'is_family', 'is_fantasy', 'is_filmnoir', 'is_history', 'is_horror', 'is_music', 'is_musical', 'is_mystery', 'is_romance', 'is_scifi', 'is_sport', 'is_thriller', 'is_war', 'is_western']].values
train_labels = data['my_points'].values

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Build the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])



# Define the optimizer
optimizer = Adam(learning_rate=0.001)

# Define the loss function
loss_function = MeanSquaredError()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])
# model.compile(optimizer='adam', loss='mean_squared_error')



# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save('movie_point_predictor_model.keras')
