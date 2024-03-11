import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from Autoencoder import *



def main():

    img_height = 32
    img_width = 32

    autoencoder = Autoencoder()
    autoencoder.build(input_shape=(None, img_height, img_width ,1))
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    autoencoder.load_weights(f"../saved_models/trained_weights_30").expect_partial()

    corrds_x = np.arange(-1, 1 + 0.08, 0.08)
    corrds_y = np.arange(-1, 1 + 0.08, 0.08)

    num_imgs_per_axis = corrds_x.shape[0]

    corrds = []
    for x in corrds_x[::-1]:
        for y in corrds_y:
            corrds.append([y,x])

    embeddings = tf.stack(corrds, axis=0)

    imgs = autoencoder.decoder(embeddings)
    imgs = tf.reshape(imgs, shape=(num_imgs_per_axis, num_imgs_per_axis, img_height, img_width, 1))
    imgs = imgs[:, :, :, :, :]

    total_img = np.zeros(shape=(num_imgs_per_axis*img_height,num_imgs_per_axis*img_width, 1), dtype=np.float32)
    
    for x in range(num_imgs_per_axis):
        for y in range(num_imgs_per_axis):
            total_img[x*img_height: x*img_height +img_height, y*img_width: y*img_width +img_width, :] = imgs[x,y, :]

    plt.figure(figsize=(10,10)) 
    plt.imshow(total_img)
    plt.axis("off")
    plt.savefig(f"../plots/GeneratedImages.png", bbox_inches='tight')
    plt.show()

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")