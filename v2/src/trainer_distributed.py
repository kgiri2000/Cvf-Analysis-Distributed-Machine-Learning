import os
import json
import socket
import tensorflow as tf
from src.data_loader import load_dataset
from src.model_builder import build_feed_forward_model
from src.utilis import save_models

def setup_tf_config(cluster_hosts=None, rank=None):

    if "TF_CONFIG" in os.environ:
        print("Using existing TF_CONFIG from environment")
        return json.loads(os.environ["TF_CONFIG"])
    
    # Local fallback
    if not cluster_hosts:
        local_ip = socket.gethostbyname(socket.gethostname())
        cluster_hosts = [f"{local_ip}:12345"]
    if rank is None:
        rank = 0

    tf_config = {
        "cluster": {"worker": cluster_hosts},
        "task": {"type": "worker", "index": rank}
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print(f"TF_CONFIG created: {tf_config}")
    return tf_config


def train_distributed(dataset_path, input_size, epochs=50, batch_size=32, 
                     learning_rate=0.001, cluster_hosts=None, rank=None):
    
    # Configure cluster
    tf_config = setup_tf_config(cluster_hosts, rank)
    worker_rank = tf_config["task"]["index"]
    num_workers = len(tf_config['cluster']['worker'])
    
    print(f"[Worker {worker_rank}] Using {num_workers} worker(s)")

    # Initialize strategy FIRST
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    print(f"[Worker {worker_rank}] Number of replicas: {strategy.num_replicas_in_sync}")

    # Load data (happens on each worker independently)
    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)
    
    # Convert to float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Per-replica batch size
    per_replica_batch = batch_size

    # Dataset creation function for distribution
    def dataset_fn(input_context):

        batch_size_per_replica = input_context.get_per_replica_batch_size(per_replica_batch)
        
        # Training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(1024, seed=42)
        train_ds = train_ds.batch(batch_size_per_replica, drop_remainder=False)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_ds = val_ds.batch(batch_size_per_replica, drop_remainder=False)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds

    # Create distributed datasets
    train_dataset = strategy.distribute_datasets_from_function(
        lambda ctx: dataset_fn(ctx)[0]
    )
    
    val_dataset = strategy.distribute_datasets_from_function(
        lambda ctx: dataset_fn(ctx)[1]
    )

    # Build model inside distributed scope
    with strategy.scope():
        model = build_feed_forward_model(
            X_train.shape[1], 
            y_train.shape[1], 
            learning_rate
        )

    # Set verbosity
    verbose = 1 if worker_rank == 0 else 0

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=verbose
    )

    # Evaluation
    loss, mae = model.evaluate(val_dataset, verbose=verbose)
    print(f"[Worker {worker_rank}] MSE = {loss:.4f}, MAE = {mae:.4f}")

    if worker_rank == 0:
        print("[Worker 0] Training complete. Model will be saved by main process.")
    
    return model, history, scaler_X