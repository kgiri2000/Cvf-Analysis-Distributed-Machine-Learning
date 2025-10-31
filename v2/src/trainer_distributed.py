import os
import json
import socket
import tensorflow as tf
from src.data_loader import load_dataset
from src.model_builder import build_feed_forward_model
from src.utils import save_models

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def setup_tf_config(cluster_hosts=None, rank=None):
    if "TF_CONFIG" in os.environ:
        print("Using existing TF_CONFIG from environment")
        return json.loads(os.environ["TF_CONFIG"])

    if not cluster_hosts:
        local_ip = socket.gethostbyname(socket.gethostname())
        cluster_hosts = [f"{local_ip}:12345"]
    if rank is None:
        rank = 0

    tf_config = {
        "cluster": {"worker": cluster_hosts},
        "task": {"type": "worker", "index": rank},
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print(f"TF_CONFIG created: {tf_config}")
    return tf_config


def train_distributed(dataset_path, input_size, epochs=50, batch_size=32, learning_rate=0.001,
                      cluster_hosts=None, rank=None):

    tf_config = setup_tf_config(cluster_hosts, rank)
    worker_rank = tf_config["task"]["index"]
    num_workers = len(tf_config["cluster"]["worker"])
    print(f"[Worker {worker_rank}] Initialized (total workers: {num_workers})")

    # Strategy must be created before any TensorFlow operations
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"[Worker {worker_rank}] Strategy initialized (replicas: {strategy.num_replicas_in_sync})")

    # Load dataset (NumPy arrays)
    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)
    X_train, X_test = X_train.astype("float32"), X_test.astype("float32")
    y_train, y_test = y_train.astype("float32"), y_test.astype("float32")

    with strategy.scope():
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], learning_rate)

    print(f"[Worker {worker_rank}] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2 if worker_rank == 0 else 0,
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Worker {worker_rank}] MSE={loss:.4f}, MAE={mae:.4f}")

    if worker_rank == 0:
        print("[Worker 0] Saving model and artifacts...")
        save_models(model, scaler_X, history, output_dir="models/distributed_worker0")
        model.save("models/feed_forward_model")
        print("[Worker 0] Done.")
    else:
        print(f"[Worker {worker_rank}] Non-chief worker finished.")

    return model, history, scaler_X
