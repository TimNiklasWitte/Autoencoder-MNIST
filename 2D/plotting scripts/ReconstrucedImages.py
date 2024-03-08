import sys
sys.path.append("../")

import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from Autoencoder import *
from Training import *



def main():

    test_ds = tfds.load("mnist", split="test", as_supervised=True)
    test_dataset = test_ds.apply(prepare_data)

    autoencoder = Autoencoder()
    autoencoder.build(input_shape=(None, 32, 32 ,1))
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    autoencoder.load_weights(f"../saved_models/trained_weights_30").expect_partial()

    num_imgs = 10

    for x in test_dataset.take(1):
        reconstruced_x = autoencoder(x)
        for i in range(num_imgs):
            img = x[i]

            fig, axes = plt.subplots(nrows=1, ncols=2)

            axes[0].imshow(x[i])
            axes[0].set_title("Input")
            axes[0].axis("off")
   
            axes[1].imshow(reconstruced_x[i])
            axes[1].set_title("Reconstructed input")
            axes[1].axis("off")

            plt.savefig(f"../plots/reconstructed images/{i}.png", bbox_inches='tight')
            plt.close()




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")