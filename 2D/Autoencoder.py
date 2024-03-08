import tensorflow as tf

from Encoder import *
from Decoder import *

class Autoencoder(tf.keras.Model):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

  
    @tf.function
    def call(self, x):
        
        embedding = self.encoder(x)
        reconstructed_x = self.decoder(embedding)

        return reconstructed_x

    @tf.function
    def train_step(self, x):

        with tf.GradientTape() as tape:
            reconstructed_x = self(x)
            loss = self.loss_function(x, reconstructed_x)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)


    def test_step(self, dataset):
          
        self.metric_loss.reset_states()
  
        for x in dataset:
            reconstructed_x = self(x)

            loss = self.loss_function(x, reconstructed_x)
            self.metric_loss.update_state(loss)


