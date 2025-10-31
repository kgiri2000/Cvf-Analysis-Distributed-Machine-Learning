import tensorflow as tf

def build_feed_forward_model(input_dim, output_dim, learning_rate=0.001):
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model