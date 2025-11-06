"""
trainer_single_gpu.py
Single-node GPU (or CPU fallback) training pipeline.
Designed to match the structure and behavior of trainer_distributed.py.
"""

import os, json, time
import tensorflow as tf
from src.data_loader import make_datasets_with_scaler
from src.model_builder import build_model, compile_model
from src.utils import ensure_dir, result

def train_single_gpu(
    dataset_path: str,
    input_size: int,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001
):
    """
    Train a model on a single GPU (or CPU fallback) with real-time per-epoch logging.
    Returns (model, history, scaler).
    """


    #Device Setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU(s) detected:")
        for g in gpus:
            print(f"   - {g.name}")
            tf.config.experimental.set_memory_growth(g, True)

        # Mixed precision for modern GPUs 
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision policy: enabled (float16 compute, float32 variables)")
    else:
        print("No GPU found — running on CPU.")


    #Dataset
    print("\nLoading and preparing dataset...")
    train_ds, val_ds, scaler = make_datasets_with_scaler(dataset_path, batch_size)


    #Model setup
    num_features = input_size - 1
    model = build_model(num_features)
    model, optimizer, loss_fn, train_loss, train_mae, val_loss, val_mae = compile_model(model, lr=learning_rate)

    print("\nModel Summary:")
    model.summary()


    # Training Loop (per-epoch logs)

    STEPS_PER_EPOCH = 100
    VAL_STEPS = 20
    train_iter, val_iter = iter(train_ds), iter(val_ds)
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    print("\nStarting training...")
    overall_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_loss.reset_state(); train_mae.reset_state()
        val_loss.reset_state(); val_mae.reset_state()

        #Train
        for _ in range(STEPS_PER_EPOCH):
            x, y = next(train_iter)
            y = tf.expand_dims(y, -1)
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                per_ex_loss = loss_fn(y, preds)
                loss = tf.reduce_mean(per_ex_loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.update_state(loss)
            train_mae.update_state(y, preds)

        #Validate
        for _ in range(VAL_STEPS):
            x, y = next(val_iter)
            y = tf.expand_dims(y, -1)
            preds = model(x, training=False)
            per_ex_loss = loss_fn(y, preds)
            val_loss.update_state(tf.reduce_mean(per_ex_loss))
            val_mae.update_state(y, preds)

        #Metrics
        tl, tm = train_loss.result().numpy(), train_mae.result().numpy()
        vl, vm = val_loss.result().numpy(), val_mae.result().numpy()
        epoch_time = time.perf_counter() - epoch_start

        #Log + Save History
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_mae"].append(tm)
        history["val_mae"].append(vm)

        print(f"Epoch {epoch+1}/{epochs},Train Loss: {tl:.6f}, MAE: {tm:.6f},Val Loss: {vl:.6f}, MAE: {vm:.6f},  Time: {epoch_time:.2f}s ")

    total_time = time.perf_counter() - overall_start


    ensure_dir("plots")

    result(history)
    print(f"Training complete in {total_time:.2f}s")
    print(f"Model + artifacts saved to models/ and plots/")

    return model, history, scaler
