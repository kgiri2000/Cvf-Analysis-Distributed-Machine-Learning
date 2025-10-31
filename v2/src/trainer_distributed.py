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


def train_distributed(dataset_path, input_size, epochs=50, batch_size=32, learning_rate=0.001,
                      cluster_hosts=None, rank=None):


    tf_config = setup_tf_config(cluster_hosts, rank)
    worker_rank = tf_config["task"]["index"]
    num_workers = len(tf_config["cluster"]["worker"])
    print(f"[Worker {worker_rank}] Using {num_workers} worker(s)")


    X_train, X_test, y_train, y_test, scaler_X = load_dataset(dataset_path, input_size)


    def make_ds(X, y, training: bool):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)


        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        return ds

    train_ds = make_ds(X_train, y_train, training=True)
    val_ds   = make_ds(X_test,  y_test,  training=False)


    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], learning_rate)


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2 if worker_rank == 0 else 0,
    )


    loss, mae = model.evaluate(val_ds, verbose=0)
    print(f"[Worker {worker_rank}] MSE={loss:.4f}, MAE={mae:.4f}")


    if worker_rank == 0:
        from src.utils import save_models  
        print("Chief node saving canonical artifacts...")
        save_models(model, scaler_X, history, output_dir="models/distributed_worker0")

    return model, history, scaler_X
