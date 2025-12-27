import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LATENT_DIM = 512
IMG_CHANNELS = 3

# -----------------------------
# Mapping Network
# -----------------------------
def mapping_network(latent_dim, layers=8):
    model = tf.keras.Sequential(name="mapping_network")
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(latent_dim, activation='relu'))
    return model


# -----------------------------
# AdaIN Layer
# -----------------------------
class AdaIN(tf.keras.layers.Layer):
    def call(self, inputs):
        content, style = inputs
        mean, variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        normalized = (content - mean) / tf.sqrt(variance + 1e-8)
        scale, bias = style
        return scale * normalized + bias


# -----------------------------
# Progressive Block
# -----------------------------
def progressive_block(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    return tf.keras.Model(inputs, x)


# -----------------------------
# Generator
# -----------------------------
def build_generator():
    z = tf.keras.Input(shape=(LATENT_DIM,))
    mapping = mapping_network(LATENT_DIM)
    w = mapping(z)

    x = tf.keras.layers.Dense(4 * 4 * 128)(w)
    x = tf.keras.layers.Reshape((4, 4, 128))(x)

    block = progressive_block((4, 4, 128))
    x = block(x)

    x = tf.keras.layers.Conv2D(
        IMG_CHANNELS, kernel_size=1, activation='tanh'
    )(x)

    return tf.keras.Model(z, x, name="generator")


# -----------------------------
# Discriminator
# -----------------------------
def build_discriminator():
    inputs = tf.keras.Input(shape=(8, 8, IMG_CHANNELS))
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, x, name="discriminator")


# -----------------------------
# Visualization
# -----------------------------
def visualize_images(images, rows, cols):
    images = (images + 1) / 2
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()


# -----------------------------
# Run Test
# -----------------------------
if __name__ == "__main__":
    gen = build_generator()
    noise = tf.random.normal([9, LATENT_DIM])
    fake_images = gen(noise)
    visualize_images(fake_images.numpy(), 3, 3)
