import pandas as pd
import os

def save_dataset(dataset, file_path, append=False):
    df = pd.DataFrame(dataset)
    if append and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

def load_dataset(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"{file_path} not found.")
