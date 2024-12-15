import os

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import ensure_directory

def load_artifacts(model_path, scaler_path):
    # Load the model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load the scaler
    with open(scaler_path, 'rb') as scaler_file:
        scaler_X = pickle.load(scaler_file)
    
    return model, scaler_X

def predict_and_analyze(true_dataset, vector_size, model, scaler_X):
    ensure_directory("datasets")
    ensure_directory("plots")
    
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
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_ar_counts.index, predicted_ar_counts.values, color='blue', label='Predicted Ar Counts')
    plt.scatter(true_ar_counts.index, true_ar_counts.values, color='red', label='True Ar Counts')
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Comparison of Predicted and True Ar Counts')
    plt.legend()
    plt.savefig('plots/comparison_counts.png')
    plt.show()
    print(f"Predictions and analysis complete. Results saved to {predicted_results_file}.")

if __name__ == "__main__":
    # Paths for the model and scaler
    model_path = "models/trained_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    # Load the saved artifacts
    model, scaler_X = load_artifacts(model_path, scaler_path)
    
    # Prediction configuration
    true_dataset = "datasets/true_dataset11.csv"
    data  = pd.read_csv(true_dataset)
    num_columns = data.shape[1]
    vector_size = num_columns
    
    # Perform prediction and analysis
    predict_and_analyze(true_dataset, vector_size, model, scaler_X)
