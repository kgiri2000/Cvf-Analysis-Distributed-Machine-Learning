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

    #Load dataset
    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)

    #initialize strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    worker_rank = tf_config["task"]["index"]


    print(f"[Worker {worker_rank}] Using {len(tf_config['cluster']['worker'])} worker(s)")

    #Build modle inside distributed scope
    with strategy.scope():
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], learning_rate)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(batch_size)
    #Train the model
    history = model.fit(
        train_dataset,
        validation_data= val_dataset,
        epochs = epochs,
        verbose=2 if worker_rank == 0 else 0
    )

    #Exaluation and save
    loss, mae = model.evaluate(val_dataset, verbose=0)
    print(f"[Worker {worker_rank}] MSE = {loss: .4f}, MAE: {mae: .4f}")

    if worker_rank == 0:
        print("Single node mode")

    return model, history, scaler_X



