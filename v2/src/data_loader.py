import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
def load_csv(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"{csv_path} not found")

def load_dataset(dataset_path, input_size, test_size= 0.2, random_state=42):
    """
    Load dataset from the CSV, split into train/test and apply normalization
    Returns: (X_train,  X_test, y_train, y_test, scaler_X)
    """
    data = load_csv(dataset_path)
    X = data.iloc[:, : input_size-1].values # First input size-1 colums are the input fratures
    y = data.iloc[:, input_size-1:].values
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler_X

    


