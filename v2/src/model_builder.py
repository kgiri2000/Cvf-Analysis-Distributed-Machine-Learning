import tensorflow as tf

def build_feed_forward_model(input_dim, output_dim, learning_rate=0.0001):
    model = tf.keras.Sequential([
        #input_dim = X_train.shape[1]
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64, activation="relu"),
        #Output layer
        tf.keras.layers.Dense(1) 
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer= optimizer, loss ='mse', metrics=['mae'])
    return model