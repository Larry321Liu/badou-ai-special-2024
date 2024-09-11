from __future__ import print_function, division

from keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import network

class Model:
    network = None
    LOG_DIR = "./logs/"

    # 私有函数
    def __init__(self):
        pass

    def train(self, epochs, batch_size=128, sample_interval=50):
        self.network = network.Network()
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.network.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.network.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.network.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.network.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.network.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.network.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.network.sample_images(epoch)
        self.network.combined.save(self.LOG_DIR + 'gan.h5')

if __name__ == '__main__':
    gan = Model()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)