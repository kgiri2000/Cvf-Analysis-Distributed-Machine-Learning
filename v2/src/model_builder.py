"""
model_builder.py
Builds and compiles the Keras model inside a strategy scope.
"""

from tensorflow.keras import layers, models, optimizers, losses, metrics

def build_model(input_dim):
    #Creates a simple feed-forward regression model.
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, dtype='float32')
    ])
    return model


def compile_model(model, lr=1e-3):
    #Compiles model with optimizer, loss, and metrics.
    optimizer = optimizers.Adam(lr)
    loss_fn = losses.MeanSquaredError(reduction=losses.Reduction.NONE)
    train_loss = metrics.Mean(name='train_loss')
    train_mae = metrics.MeanAbsoluteError(name='train_mae')
    val_loss = metrics.Mean(name='val_loss')
    val_mae = metrics.MeanAbsoluteError(name='val_mae')

    return model, optimizer, loss_fn, train_loss, train_mae, val_loss, val_mae
