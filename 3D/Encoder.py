import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self):
 
        super(Encoder, self).__init__()

        self.layer_list = [
            # (batch_size, 32, 32, 1)
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 16, 16, 8)

            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 8, 8, 16)

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 4, 4, 32)

            tf.keras.layers.Flatten(),
            # (batch_size, 512)

            tf.keras.layers.Dense(3, activation='tanh')
        ]

    
    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)

        return x
    
    
