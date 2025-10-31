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


def make_dataset(X, y, batch_size=32, training=False):
    # Enforce correct dtype to prevent PerReplica mismatch
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Enable auto-sharding for multi-worker
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    return ds


def train_distributed(dataset_path, input_size, epochs=50, batch_size=32, learning_rate=0.001,
                      cluster_hosts=None, rank=None):

    # Configure cluster
    tf_config = setup_tf_config(cluster_hosts, rank)
    worker_rank = tf_config["task"]["index"]
    num_workers = len(tf_config["cluster"]["worker"])
    print(f"[Worker {worker_rank}] Initialized (total workers: {num_workers})")


    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"[Worker {worker_rank}] Strategy initialized (replicas: {strategy.num_replicas_in_sync})")


    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)


    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Build and compile model inside strategy scope
    with strategy.scope():
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], learning_rate)

    # Build datasets AFTER strategy creation
    train_ds = make_dataset(X_train, y_train, batch_size=batch_size, training=True)
    val_ds = make_dataset(X_test, y_test, batch_size=batch_size, training=False)

    print(f"[Worker {worker_rank}] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2 if worker_rank == 0 else 0
    )

    # Evaluate
    loss, mae = model.evaluate(val_ds, verbose=0)
    print(f"[Worker {worker_rank}] MSE={loss:.4f}, MAE={mae:.4f}")

    # Save (chief only)
    if worker_rank == 0:
        print("[Worker 0] Saving model and training artifacts...")
        save_models(model, scaler_X, history, output_dir="models/distributed_worker0")
        model.save("models/feed_forward_model")
        print("[Worker 0] Model saved successfully.")
    else:
        print(f"[Worker {worker_rank}] Non-chief worker done (no save).")

    return model, history, scaler_X
