import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import ensure_dir

def load_model_and_scalar(model_dir, scaler_path):
    model_dir = "models/feed_forward_model"
    scaler_path= "models/scaler.pkl"


    #Load trained Tensorflow model and scalar from the disk

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scalar file not found: {scaler_path}")
    
    model = tf.keras.models.load_model(model_dir)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully")

    return model, scaler


def predict_and_analyze(true_dataset, vector_size, model, scaler_X, training_type):

    # Load the dataset for prediction
    predict_file_path = pd.read_csv(true_dataset)
    to_predict_file = predict_file_path.iloc[:, :vector_size - 1].values  # Input features
    to_predict_file_scaled = scaler_X.transform(to_predict_file)
    
    # Make predictions
    predictions = model.predict(to_predict_file_scaled)
    predict_file_path["Ar"] = predictions
    to_predict_node = ''.join(filter(str.isdigit, os.path.splitext(true_dataset)[0]))
    predicted_results_file = f"datasets/predict_dataset{to_predict_node}_using{vector_size - 2}.csv"
    predict_file_path.to_csv(predicted_results_file, index=False)
    
    # Analyze predictions
    predicted_data = pd.read_csv(predicted_results_file)
    predicted_data['Ar'] = predicted_data['Ar'].apply(lambda x: int(x))
    ar_counts = predicted_data['Ar'].value_counts()
    ar_counts_df = ar_counts.reset_index()
    ar_counts_df.columns = ["Ar", "Count"]
    ar_counts_df.to_csv(f'datasets/predicted_ar_counts{to_predict_node}_using{vector_size-2}.csv', index=False)
    
    # Load true data
    true_data = pd.read_csv(true_dataset)
    true_data.rename(columns={true_data.columns[-1]: 'Ar'}, inplace=True)
    
    # Compute counts
    predicted_ar_counts = ar_counts
    true_ar_counts = true_data['Ar'].value_counts()

    ensure_dir("plots")
    
    # Plot comparison
    plt_name = f'comparison_count_{training_type}_for_{to_predict_node}_node_using_{vector_size-2}.png'
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_ar_counts.index, predicted_ar_counts.values, color='blue', label='Predicted Ar Counts')
    plt.scatter(true_ar_counts.index, true_ar_counts.values, color='red', label='True Ar Counts')
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Comparison of Predicted and True Ar Counts')
    plt.legend()
    plt.savefig(f'plots/{plt_name}')
    plt.show()
    print(f"Predictions and analysis complete. Results saved to {predicted_results_file}.")