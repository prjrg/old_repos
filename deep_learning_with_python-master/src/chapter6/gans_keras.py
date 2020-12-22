
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization, Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

def build_generator(latent_dim: int):

    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(np.prod((28, 28, 1)), activation='tanh'),
        Reshape((28, 28, 1))
    ])

    model.summary()

    z = Input(shape=(latent_dim,))
    generated = model(z)

    return Model(z, generated)


def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28,28,1)),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ], name='discriminator')

    model.summary()

    image = Input(shape=(28,28, 1))
    output = model(image)

    return Model(image, output)


def train(generator, discriminator, combined, steps, batch_size):
    (x_train, _), _ = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    latent_dim = generator.input_shape[1]

    for step in range(steps):
        real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)
        discriminator_real_loss = discriminator.train_on_batch(real_images, real)
        discriminator_fake_loss = discriminator.train_on_batch(generated_images, fake)
        discriminator_loss = 0.5 * np.add(discriminator_real_loss, discriminator_fake_loss)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        generator_loss = combined.train_on_batch(noise, real)

        print("%d [Discriminator loss: %.4f%%, acc.: %.2f%%] [Generator loss: %.4f%%]" %
              (step, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))


def plot_generated_images(generator):
    """
    Display a 2D plot of the generated images.
    We only need the decoder, because we'll manually sample the distribution z
    :param decoder: the decoder network
    """

    # display a nxn 2D manifold of digits
    n = 10
    digit_size = 28

    figure = np.zeros((digit_size * n, digit_size * n))

    latent_dim = generator.input_shape[1]

    noise = np.random.normal(0, 1, (n * n, latent_dim))

    generated_images = generator.predict(noise)

    for i in range(n):
        for j in range(n):
            slice_i = slice(i * digit_size, (i + 1) * digit_size)
            slice_j = slice(j * digit_size, (j + 1) * digit_size)
            figure[slice_i, slice_j] = np.reshape(generated_images[i * n + j], (28, 28))

    # plot the results
    plt.figure(figsize=(6, 5))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


latent_dim = 64
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)

z = Input(shape=(latent_dim,))
generated_image = generator(z)

discriminator.trainable = False

real_or_fake = discriminator(generated_image)

combined = Model(z, real_or_fake)
combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002, beta_1=0.5))

train(generator=generator, discriminator=discriminator, combined=combined, steps=15000, batch_size=128)
plot_generated_images(generator)