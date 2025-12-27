import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Load and preprocess MNIST
# ===============================
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

input_shape = (28, 28, 1)
latent_dim = 2
epochs = 10
batch_size = 128

# ===============================
# Encoder (Requirement Given)
# ===============================
def build_encoder(input_shape, latent_dim):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)

    return tf.keras.Model(inputs, [z_mean, z_log_var], name="Encoder")

# ===============================
# Sampling Layer
# ===============================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ===============================
# Decoder
# ===============================
def build_decoder(latent_dim):
    inputs = tf.keras.layers.Input(shape=(latent_dim,))

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(28 * 28, activation='sigmoid')(x)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(x)

    return tf.keras.Model(inputs, outputs, name="Decoder")

# ===============================
# VAE Model with Loss
# ===============================
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z)

        # Reconstruction Loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed),
                axis=(1, 2)
            )
        )

        # KL Divergence Loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)

        return reconstructed

# ===============================
# Build and Train VAE
# ===============================
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim)

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())

print("Training VAE...")
vae.fit(
    x_train,
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, x_test)
)

# ===============================
# Visualize Reconstructions
# ===============================
decoded_images = vae.predict(x_test[:10])

plt.figure(figsize=(10, 4))
for i in range(10):
    # Original
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_images[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.suptitle("Top: Original Images | Bottom: Reconstructed Images")
plt.show()
