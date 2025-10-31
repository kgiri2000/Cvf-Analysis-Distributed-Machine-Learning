import os
import json
import socket
import tensorflow as tf
from src.data_loader import load_dataset
from src.model_builder import build_feed_forward_model
from src.utilis import save_models  

# Optional: reduce TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all logs, 1 = warning, 2 = error, 3 = fatal


def setup_tf_config(cluster_hosts=None, rank=None):

    if "TF_CONFIG" in os.environ:
        print("Using existing TF_CONFIG from environment")
        return json.loads(os.environ["TF_CONFIG"])

    # Local fallback (for testing)
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

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Auto-shard configuration (important for multi-worker)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    return ds


def train_distributed(dataset_path, input_size, epochs=50, batch_size=32, learning_rate=0.001,
                      cluster_hosts=None, rank=None):



    tf_config = setup_tf_config(cluster_hosts, rank)
    worker_rank = tf_config["task"]["index"]
    num_workers = len(tf_config["cluster"]["worker"])
    print(f"[Worker {worker_rank}] Initialized (total workers: {num_workers})")

 
    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)


    train_ds = make_dataset(X_train, y_train, batch_size=batch_size, training=True)
    val_ds = make_dataset(X_test, y_test, batch_size=batch_size, training=False)

 
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    print(f"[Worker {worker_rank}] Strategy initialized, starting build inside scope...")


    with strategy.scope():
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], learning_rate)

    print(f"[Worker {worker_rank}] Model compiled successfully. Starting training...")


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2 if worker_rank == 0 else 0,  # only chief logs epochs
    )

    print(f"[Worker {worker_rank}] Training complete. Evaluating model...")


    loss, mae = model.evaluate(val_ds, verbose=0)
    print(f"[Worker {worker_rank}] Final Metrics → MSE={loss:.4f}, MAE={mae:.4f}")


    if worker_rank == 0:
        print("[Worker 0] Chief node saving trained model and scaler...")
        save_models(model, scaler_X, history, output_dir="models/distributed_worker0")
        print("[Worker 0] Artifacts saved successfully.")
    else:
        print(f"[Worker {worker_rank}] Worker finished without saving (non-chief).")

    print(f"[Worker {worker_rank}] Training job finished successfully.")
    return model, history, scaler_X
