import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_models(model, scaler, history, output_dir= "models"):

    ensure_dir(output_dir)

    #Save tensorflow model
    with open(os.path.join(output_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)

    #Save scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler,f)
    
    #Save training history
    with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)
    
    print(f"Artifacts saved to {output_dir}")

def load_models(model_path, scaler_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    #Load the scaler
    with open(scaler_path, 'rb') as scaler_file:
        scaler_X = pickle.load(scaler_file)
    
    return model, scaler_X

def result(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/evaluation_plot.png')

