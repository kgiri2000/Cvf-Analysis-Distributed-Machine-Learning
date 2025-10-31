import os
import json
import socket
import tensorflow as tf
from src.data_loader import load_dataset
from src.model_builder import build_feed_forward_model
from src.utilis import save_models

def setup_tf_config(cluster_hosts=None, rank=None):
    #Automatically set TF_CONFIG
    #If not provided, defaults to the single-node setup

    if "TF_CONFIG" in os.environ:
        print("Using existing TF_CONFIG from environment")
        return json.loads(os.environ["TF_CONFIG"])
    
    #Local fall back
    if not cluster_hosts:
        local_ip = socket.gethostbyname(socket.gethostname())
        cluster_hosts = [f"{local_ip}:12345"]
    if rank is None:
        rank=0

    tf_config = {
        "cluster": {"worker": cluster_hosts},
        "task": {"type": "worker", "index":rank}
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print(f"TF_CONFIG created: {tf_config}")
    return tf_config

def train_distributed(dataset_path, input_size, epochs=50, batch_size = 32, learning_rate=0.001, cluster_hosts= None, rank=None):
    #Distributed feed-forward traning using MultiWorkerMirrorStrategy

    #Configure cluster
    tf_config = setup_tf_config(cluster_hosts, rank)

    # #Load dataset
    # X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)

    #initialize strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    worker_rank = tf_config["task"]["index"]
    per_worker_batch_size = batch_size

    print(f"[Worker {worker_rank}] Using {len(tf_config['cluster']['worker'])} worker(s)")

    #Build modle inside distributed scope
    def dataset_fn(input_context):

        # Load data (this happens on each worker)
        X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)
        
        batch_size_per_replica = batch_size // input_context.num_input_pipelines
        
        # Training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shard(
            num_shards=input_context.num_input_pipelines,
            index=input_context.input_pipeline_id
        )
        train_ds = train_ds.shuffle(1024)
        train_ds = train_ds.batch(batch_size_per_replica)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_ds = val_ds.shard(
            num_shards=input_context.num_input_pipelines,
            index=input_context.input_pipeline_id
        )
        val_ds = val_ds.batch(batch_size_per_replica)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds

    # Load data once to get scaler (on chief worker)
    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)

    # Create distributed datasets
    options = tf.distribute.InputOptions(
        experimental_fetch_to_device=True,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER
    )
    
    train_dataset = strategy.distribute_datasets_from_function(
        lambda ctx: dataset_fn(ctx)[0], options=options
    )
    val_dataset = strategy.distribute_datasets_from_function(
        lambda ctx: dataset_fn(ctx)[1], options=options
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


        

    #Train the model
    history = model.fit(
        train_dataset,
        validation_data= val_dataset,
        epochs = epochs,
        verbose=verbose
    )

    #Exaluation and save
    loss, mae = model.evaluate(val_dataset, verbose=verbose)
    print(f"[Worker {worker_rank}] MSE = {loss: .4f}, MAE: {mae: .4f}")

    if worker_rank == 0:
        print("Single node mode")

    return model, history, scaler_X



