import tensorflow as tf

class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(3,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, X):
        return self.net(X)