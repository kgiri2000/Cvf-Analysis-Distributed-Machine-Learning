"""
data_loader.py
Loads CSV data, splits into train/val, applies StandardScaler normalization,
and returns TensorFlow datasets + the fitted scaler.
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def make_datasets_with_scaler(data_file, global_batch_size, train_split=0.8):
    """
    Reads CSV, normalizes features with StandardScaler,
    and returns (train_ds, val_ds, scaler).
    """
    #Load CSV into pandas
    df = pd.read_csv(data_file)
    
    X = df.iloc[:, :-1].values.astype("float32")  # Features
    y = df.iloc[:, -1].values.astype("float32")   # Labels

    #Split into train/val
    split_idx = int(len(X) * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    num_train = len(X_train)
    num_val = len(X_val)
    steps_per_epoch = num_train // global_batch_size
    val_steps = num_val // global_batch_size

    #Fit StandardScaler on train features only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    #Convert to TensorFlow Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    #Shuffle, batch, repeat, prefetch for performance
    train_ds = (train_ds.shuffle(10000)
                        .batch(global_batch_size)
                        .repeat()
                        .prefetch(tf.data.AUTOTUNE))
    val_ds = (val_ds.batch(global_batch_size)
                    .repeat()
                    .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, scaler, steps_per_epoch, val_steps
