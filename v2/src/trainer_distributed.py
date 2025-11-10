"""
trainer_distributed.py
Distributed TensorFlow training with per-epoch logging and timing breakdown.
"""

import os, json, time
import tensorflow as tf
from src.data_loader import make_datasets_with_scaler
from src.model_builder import build_model, compile_model
from src.utils import ensure_dir, result


def train_distributed(
    dataset_path: str,
    input_size: int,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    cluster_hosts: list[str] = None,
    rank: int = 0
):
    """Run multi-worker distributed training with per-phase timing."""

    #Environment & role detection

    tf_conf = os.environ.get("TF_CONFIG")
    if not tf_conf:
        raise RuntimeError("TF_CONFIG not set (must be run via launcher or with env set).")

    task = json.loads(tf_conf)["task"]
    task_type = task.get("type", "worker")
    task_index = task.get("index", 0)
    worker_id = f"[{task_type}:{task_index}]"
    is_chief = (task_type == "chief")

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"{worker_id} Initialized with {strategy.num_replicas_in_sync} replicas", flush=True)

    #Data preparation
    PER_REPLICA_BS = batch_size
    GLOBAL_BS = PER_REPLICA_BS * strategy.num_replicas_in_sync

    data_start = time.perf_counter()
    train_ds, val_ds, scaler, steps_per_epoch, steps_per_vals = make_datasets_with_scaler(
        dataset_path, GLOBAL_BS
    )
    data_time = time.perf_counter() - data_start

    dist_train = strategy.experimental_distribute_dataset(train_ds)
    dist_val = strategy.experimental_distribute_dataset(val_ds)


    #Model setup

    num_features = input_size - 1
    with strategy.scope():
        model = build_model(num_features)
        model, optimizer, loss_fn, train_loss, train_mae, val_loss, val_mae = compile_model(
            model, lr=learning_rate
        )

        @tf.function
        def train_step(batch):
            x, y = batch
            y = tf.expand_dims(y, -1)
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                per_ex_loss = loss_fn(y, preds)
                loss = tf.nn.compute_average_loss(per_ex_loss, global_batch_size=GLOBAL_BS)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.update_state(loss)
            train_mae.update_state(y, preds)

        @tf.function
        def val_step(batch):
            x, y = batch
            y = tf.expand_dims(y, -1)
            preds = model(x, training=False)
            per_ex_loss = loss_fn(y, preds)
            loss = tf.nn.compute_average_loss(per_ex_loss, global_batch_size=GLOBAL_BS)
            val_loss.update_state(loss)
            val_mae.update_state(y, preds)

        @tf.function
        def distributed_train_step(batch):
            strategy.run(train_step, args=(batch,))

        @tf.function
        def distributed_val_step(batch):
            strategy.run(val_step, args=(batch,))


    #Training loop with phase timing
   
    EPOCHS = epochs
    STEPS_PER_EPOCH = steps_per_epoch
    VAL_STEPS = steps_per_vals

    train_iter, val_iter = iter(dist_train), iter(dist_val)
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    if is_chief:
        print(f"Data prep time: {data_time:.2f}s")

    overall_start = time.perf_counter()

    for epoch in range(EPOCHS):
        train_loss.reset_state()
        train_mae.reset_state()
        val_loss.reset_state()
        val_mae.reset_state()

        epoch_start = time.perf_counter()

        #TRAIN PHASE
        compute_start = time.perf_counter()
        for _ in range(STEPS_PER_EPOCH):
            distributed_train_step(next(train_iter))
        compute_time = time.perf_counter() - compute_start

        #VALIDATION
        val_start = time.perf_counter()
        for _ in range(VAL_STEPS):
            distributed_val_step(next(val_iter))
        val_time = time.perf_counter() - val_start

        #METRICS
        tl, tm = train_loss.result().numpy(), train_mae.result().numpy()
        vl, vm = val_loss.result().numpy(), val_mae.result().numpy()
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_mae"].append(tm)
        history["val_mae"].append(vm)

        if is_chief:
            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"| TrainLoss {tl:.4f}, MAE {tm:.4f} "
                f"| ValLoss {vl:.4f}, MAE {vm:.4f} "
                f"| Times: total {epoch_time:.2f}s (compute {compute_time:.2f}s, val {val_time:.2f}s)",
                flush=True,
            )

    total_time = time.perf_counter() - overall_start


    #Save artifacts (chief only)
 
    if is_chief:
        ensure_dir("plots")
        result(history)
        print(f"\nTraining complete in {total_time:.2f}s")

    return model, history if is_chief else None, scaler, total_time
