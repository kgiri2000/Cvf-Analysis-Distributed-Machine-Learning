import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def feed_forward(dataset_path):
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :12].values  # First 12 columns are the input features
    y = data.iloc[:, 12:].values  # Last 2 columns are the output labels
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer
        tf.keras.layers.Dense(2)  # Output layer with 2 neurons (since we have 2 output labels)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
    return model, history