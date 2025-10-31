import os
import json
import socket
import tensorflow as tf
from src.data_loader import load_dataset
from src.model_builder import build_feed_forward_model
from src.utilis import save_models

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


def make_dataset_fn(X, y, batch_size, is_training=True):
    def dataset_fn(input_context):
        per_replica_batch = input_context.get_per_replica_batch_size(batch_size)

        # Create dataset *outside* the graph (from tensors, not numpy arrays)
        X_local = tf.convert_to_tensor(X, dtype=tf.float32)
        y_local = tf.convert_to_tensor(y, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((X_local, y_local))

        ds = ds.shard(
            num_shards=input_context.num_input_pipelines,
            index=input_context.input_pipeline_id,
        )

        if is_training:
            ds = ds.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        ds = ds.batch(per_replica_batch, drop_remainder=is_training)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return dataset_fn


def train_distributed(dataset_path, input_size, epochs=50, batch_size=32, learning_rate=0.001,
                      cluster_hosts=None, rank=None):

    tf_config = setup_tf_config(cluster_hosts, rank)
    worker_rank = tf_config["task"]["index"]
    num_workers = len(tf_config["cluster"]["worker"])
    print(f"[Worker {worker_rank}] Initialized (total workers: {num_workers})")

    # Strategy setup
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"[Worker {worker_rank}] Strategy initialized (replicas: {strategy.num_replicas_in_sync})")

    # Load data on each worker (same file, TF will shard automatically)
    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)
    X_train, X_test = X_train.astype("float32"), X_test.astype("float32")
    y_train, y_test = y_train.astype("float32"), y_test.astype("float32")
    print(f"[Worker {worker_rank}] Dataset loaded: X_train={X_train.shape}, y_train={y_train.shape}")

    # Proper distributed datasets
    train_ds = strategy.distribute_datasets_from_function(
        make_dataset_fn(X_train, y_train, batch_size, is_training=True)
    )
    val_ds = strategy.distribute_datasets_from_function(
        make_dataset_fn(X_test, y_test, batch_size, is_training=False)
    )

    with strategy.scope():
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], learning_rate)

    print(f"[Worker {worker_rank}] Starting distributed training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2 if worker_rank == 0 else 0,
    )

    loss, mae = model.evaluate(val_ds, verbose=0)
    print(f"[Worker {worker_rank}] Final Results -> MSE={loss:.4f}, MAE={mae:.4f}")

    if worker_rank == 0:
        print("[Worker 0] Saving model and artifacts...")
        save_models(model, scaler_X, history, output_dir="models/distributed_worker0")
        model.save("models/feed_forward_model")
        print("[Worker 0] Model saved successfully")
    else:
        print(f"[Worker {worker_rank}] Non-chief worker finished.")

    return model, history, scaler_X
