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
        pickle.dump(history, f)
    
    print(f"Artifacts saved to {output_dir}")

def load_models(model_path, scaler_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    #Load the scaler
    with open(scaler_path, 'rb') as scaler_file:
        scaler_X = pickle.load(scaler_file)
    
    return model, scaler_X

def result(history, output_dir="plots"):
    #Plot loss and MAE curves and save as PNG.
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (log scale)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.yscale('log'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    print("Saved loss curve to plots/loss_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE (log scale)')
    plt.xlabel('Epoch'); plt.ylabel('MAE')
    plt.yscale('log'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_curve.png"))
    print("Saved MAE curve to plots/mae_curve.png")