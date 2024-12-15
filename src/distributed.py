import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def feed_forward_distributed(dataset_path, input_size):
    # Load and preprocess data
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :input_size-1].values  # First input size-1 columns are the input features
    y = data.iloc[:, input_size-1:].values  # Last column is the output labels
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Use MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    # Define model within the strategy scope
    with strategy.scope():
        learning_rate = 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define the neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
            tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
            tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer
            tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer
            tf.keras.layers.Dense(1)  # Output layer
        ])

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/distributed_evaluation_plot.png')

    return model, history, scaler_X