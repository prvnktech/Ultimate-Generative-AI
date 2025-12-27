import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

LATENT_DIM = 512
IMAGE_SIZE = 32
BATCH_SIZE = 16
EPOCHS = 5

# =========================
# Mapping Network
# =========================
def mapping_network(latent_dim, layers=8):
    model = tf.keras.Sequential(name="MappingNetwork")
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(latent_dim, activation='relu'))
    return model

mapping_net = mapping_network(LATENT_DIM)

# =========================
# AdaIN Layer
# =========================
class AdaIN(tf.keras.layers.Layer):
    def call(self, inputs):
        content, style = inputs
        mean, variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        normalized = (content - mean) / tf.sqrt(variance + 1e-8)
        scale, bias = style
        return scale * normalized + bias

# =========================
# Generator Block
# =========================
def progressive_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

# =========================
# Generator
# =========================
def build_generator():
    noise = tf.keras.Input(shape=(LATENT_DIM,))
    style = mapping_net(noise)

    x = tf.keras.layers.Dense(4 * 4 * 128)(style)
    x = tf.keras.layers.Reshape((4, 4, 128))(x)

    x = progressive_block(x, 128)  # 8x8
    x = progressive_block(x, 64)   # 16x16
    x = progressive_block(x, 32)   # 32x32 ✅

    output = tf.keras.layers.Conv2D(3, 1, activation='sigmoid')(x)
    return tf.keras.Model(noise, output, name="Generator")

    noise = tf.keras.Input(shape=(LATENT_DIM,))
    style = mapping_net(noise)

    x = tf.keras.layers.Dense(4 * 4 * 128)(style)
    x = tf.keras.layers.Reshape((4, 4, 128))(x)

    x = progressive_block(x, 128)
    x = progressive_block(x, 64)

    output = tf.keras.layers.Conv2D(3, 1, activation='sigmoid')(x)
    return tf.keras.Model(noise, output, name="Generator")

# =========================
# Discriminator
# =========================
def build_discriminator():
    img = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation='relu')(img)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(img, output, name="Discriminator")

generator = build_generator()
discriminator = build_discriminator()

# =========================
# Training Setup
# =========================
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Dummy dataset (replace with real images)
def get_dataset():
    data = np.random.rand(200, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")
    return tf.data.Dataset.from_tensor_slices(data).shuffle(100).batch(BATCH_SIZE)

dataset = get_dataset()

# =========================
# Training Step
# =========================
@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        d_loss = loss_fn(tf.ones_like(real_output), real_output) + \
                 loss_fn(tf.zeros_like(fake_output), fake_output)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    return g_loss, d_loss

# =========================
# Train Loop + TensorBoard
# =========================
log_dir = "logs/stylegan_tf"
summary_writer = tf.summary.create_file_writer(log_dir)

for epoch in range(EPOCHS):
    for step, real_images in enumerate(dataset):
        g_loss, d_loss = train_step(real_images)

    with summary_writer.as_default():
        tf.summary.scalar("Generator Loss", g_loss, step=epoch)
        tf.summary.scalar("Discriminator Loss", d_loss, step=epoch)

    print(f"Epoch {epoch+1}: G={g_loss:.4f}, D={d_loss:.4f}")

# =========================
# Visualization
# =========================
def visualize_images(images, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()

sample_noise = tf.random.normal([9, LATENT_DIM])
generated_images = generator(sample_noise)
visualize_images(generated_images, 3, 3)
