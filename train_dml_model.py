import os
import pickle
from src.distributed import feed_forward_distributed
from src.utils import ensure_directory

def train_and_save_model(file_path, vector_size, model_path, scaler_path, history_path):
    ensure_directory("models")
    # Train the model
    model, history, scaler_X = feed_forward_distributed(file_path, vector_size)
    
    # Save the model
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    # Save the scaler
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler_X, scaler_file)
    
    # Optionally save the training history
    with open(history_path, 'wb') as history_file:
        pickle.dump(history.history, history_file)

    print(f"Model, scaler, and training history saved successfully.")

if __name__ == "__main__":
    # Training configuration
    file_path = "datasets/dataset3_10_11.csv"
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    last_number = parts[-1].split('.')[0]
    vector_size = int(last_number) + 2

    # Paths to save the model and artifacts
    model_path = "models/trained_model.pkl"
    scaler_path = "models/scaler.pkl"
    history_path = "models/training_history.pkl"

    # Train and save the model
    train_and_save_model(file_path, vector_size, model_path, scaler_path, history_path)
