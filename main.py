import os
from src.dijkstra import Dijkstra
from src.dataset import save_dataset
from src.feed_forward import feed_forward
from src.utils import ensure_directory
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ensure_directory("results")
    ensure_directory("graphs")

    # Generate dataset
    file_path = "results/dataset.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

    all_graphs = [f"graphs/dijkstra_{i}.txt" for i in range(3, 11)]
    for graph_path in all_graphs:
        result_path = "results"
        cvf = Dijkstra(graph_path, result_path)
        cvf.analyse()
        save_dataset(cvf.dataset, file_path, append=True)

    # Train machine learning model
    model, history = feed_forward(file_path)
    predict_file_path = pd.read_csv("dataset11.csv")
    predict_file = predict_file_path.iloc[:, :12].values  # First 12 columns are the input features
    #predict_file_scaled = scaler_X.fit_transform(predict_file)
    predictions = model.predict(predict_file)
    Ar_predictions = predictions[:, 0]
    M_predictions = predictions[:, 1]
    predict_file_path["Ar"] = Ar_predictions
    predict_file_path["M"] = M_predictions
    predict_file_path.to_csv("dataset_predict11.csv", index=False)
    
    
    
    # Load the predicted results from the CSV file
    predicted_results_file = 'dataset_predict11.csv'
    predicted_data = pd.read_csv(predicted_results_file)
    # Convert to int and ensure negative values are replaced with their absolute values
    predicted_data['Ar'] = predicted_data['Ar'].apply(lambda x: abs(int(x)))  
    predicted_data['M'] = predicted_data['M'].apply(lambda x: abs(int(x)))

    # predicted_data['Ar'] = predicted_data['Ar'].apply(lambda x: max(0, int(x)))  # Convert to int, and if < 0, set to 0
    # predicted_data['M'] = predicted_data['M'].apply(lambda x: max(0, int(x)))    # Convert to int, and if < 0, set to 0

    #Count Ar and M and plot graph
    ar_counts = predicted_data['Ar'].value_counts()
    m_counts = predicted_data['M'].value_counts()
    
    # Filter 'Ar' counts to remove values with count > 25000
    ar_counts = ar_counts[ar_counts <= 2000]

    # Filter 'M' counts to remove values with count > 25000
    m_counts = m_counts[m_counts <= 2000]


    # Create the dot plot
    plt.figure(figsize=(10, 6))

    # Plot dots for 'Ar'
    plt.scatter(ar_counts.index, ar_counts.values, color='blue', label='Ar Counts')

    # Plot dots for 'M'
    plt.scatter(m_counts.index, m_counts.values, color='red', label='M Counts')

    # Add labels and legend
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Counts of Ar and M Ranks')
    plt.savefig('plots/predicted_counts.png')
    plt.legend()
    
    # Load the predicted results from the CSV file
    true_results_file = 'dataset11.csv'
    true_data = pd.read_csv(true_results_file)

    #Count Ar and M and plot graph
    true_data.columns.values[-2] = 'Ar'
    true_data.columns.values[-1] = 'M'
    ar_counts = true_data['Ar'].value_counts()
    m_counts = true_data['M'].value_counts()

    # Create the dot plot
    plt.figure(figsize=(10, 6))

    # Plot dots for 'Ar'
    plt.scatter(ar_counts.index, ar_counts.values, color='blue', label='Ar Counts')
    # Plot dots for 'M'
    plt.scatter(m_counts.index, m_counts.values, color='red', label='M Counts')
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Counts of Ar and M Ranks')
    plt.savefig('plots/true_counts.png')
    plt.legend()

        
        

