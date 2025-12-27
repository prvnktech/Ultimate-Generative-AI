import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

# Set page configuration
st.set_page_config(page_title="GAN MNIST Generator (TensorFlow)", layout="wide", page_icon="🎨")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("# 🎨 GAN MNIST Generator (TensorFlow)")
st.markdown("### Generate handwritten digits using TensorFlow-based GAN")

# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16
OUTPUT_DIR = "generated_images"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generator Model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                               padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                               padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                               padding="same", use_bias=False,
                               activation="tanh")
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2),
                      padding="same", input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2),
                      padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Training step
@tf.function
def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    gen_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    disc_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'discriminator' not in st.session_state:
    st.session_state.discriminator = None
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎨 Generate", "🏋️ Train", "💻 Code"])

# Tab 1: Generate
with tab1:
    st.markdown("### 🎨 Generate Images")
    
    if st.session_state.generator is None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("👆 No trained model available. Please train the model first in the 'Train' tab or load existing images from the 'generated_images' folder.")
        
        with col2:
            # Check if generated images exist
            if os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
                st.success(f"Found {len(os.listdir(OUTPUT_DIR))} generated images")
                if st.button("📂 View Generated Images"):
                    st.session_state.view_images = True
        
        # Display existing images if available
        if 'view_images' in st.session_state and st.session_state.view_images:
            st.markdown("### 📸 Existing Generated Images")
            image_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
            
            if image_files:
                selected_epoch = st.select_slider(
                    "Select Epoch",
                    options=image_files,
                    value=image_files[-1]
                )
                
                img_path = os.path.join(OUTPUT_DIR, selected_epoch)
                st.image(img_path, caption=f"Generated Images - {selected_epoch}", use_container_width=True)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Controls")
            num_images = st.slider("Number of images to generate", 4, 25, 16)
            
            if st.button("✨ Generate New Images", type="primary"):
                with st.spinner("Generating images..."):
                    # Generate images
                    noise = tf.random.normal([num_images, NOISE_DIM])
                    generated_images = st.session_state.generator(noise, training=False)
                    
                    st.session_state.generated_images = generated_images
                
                st.success(f"✅ Generated {num_images} images!")
        
        with col2:
            st.markdown("#### Generated Output")
            
            if 'generated_images' in st.session_state:
                generated_images = st.session_state.generated_images
                
                # Create matplotlib figure
                grid_size = int(np.ceil(np.sqrt(len(generated_images))))
                fig = plt.figure(figsize=(10, 10))
                
                for i in range(len(generated_images)):
                    plt.subplot(grid_size, grid_size, i + 1)
                    plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
                    plt.axis("off")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("👆 Click 'Generate New Images' to create images")

# Tab 2: Train
with tab2:
    st.markdown("### 🏋️ Train GAN Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Training Configuration")
        
        epochs = st.slider("Number of Epochs", 1, 100, 10)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}"
        )
        
        if st.button("🚀 Start Training", type="primary"):
            st.session_state.training_started = True
            
            with st.spinner("Loading dataset..."):
                # Load and prepare dataset
                (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
                
                train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
                train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
                
                dataset = (
                    tf.data.Dataset.from_tensor_slices(train_images)
                    .shuffle(BUFFER_SIZE)
                    .batch(BATCH_SIZE, drop_remainder=True)
                )
            
            # Initialize models
            generator = build_generator()
            discriminator = build_discriminator()
            
            generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            
            # Fixed noise for visualization
            seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
            
            # Progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_metrics = st.empty()
            
            # Training loop
            for epoch in range(1, epochs + 1):
                status_text.text(f"Training Epoch {epoch}/{epochs}...")
                
                for image_batch in dataset:
                    gen_loss, disc_loss = train_step(
                        image_batch, generator, discriminator,
                        generator_optimizer, discriminator_optimizer
                    )
                
                # Save images every 5 epochs or last epoch
                if epoch % 5 == 0 or epoch == epochs:
                    predictions = generator(seed, training=False)
                    
                    fig = plt.figure(figsize=(4, 4))
                    for i in range(predictions.shape[0]):
                        plt.subplot(4, 4, i + 1)
                        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
                        plt.axis("off")
                    
                    path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:04d}.png")
                    plt.savefig(path)
                    plt.close(fig)
                
                # Update progress
                progress_bar.progress(epoch / epochs)
                epoch_metrics.metric(
                    f"Epoch {epoch}",
                    f"G Loss: {gen_loss:.4f} | D Loss: {disc_loss:.4f}"
                )
            
            # Save models to session state
            st.session_state.generator = generator
            st.session_state.discriminator = discriminator
            st.session_state.current_epoch = epochs
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ Training complete! Trained for {epochs} epochs.")
            st.balloons()
    
    with col2:
        st.markdown("#### Training Status")
        
        if st.session_state.training_started:
            st.success(f"✅ Model trained for {st.session_state.current_epoch} epochs")
            
            if os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
                st.markdown("##### Latest Generated Images")
                image_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
                if image_files:
                    latest_image = os.path.join(OUTPUT_DIR, image_files[-1])
                    st.image(latest_image, caption="Latest Generation", use_container_width=True)
        else:
            st.info("⏳ Training not started")
        
        st.markdown("---")
        st.markdown("#### About")
        st.markdown("""
        This GAN uses:
        - **Generator**: Conv2DTranspose layers for upsampling
        - **Discriminator**: Conv2D layers for classification
        - **Training**: Binary cross-entropy loss with Adam optimizer
        - **Dataset**: MNIST handwritten digits
        """)

# Tab 3: Code
with tab3:
    st.markdown("### 💻 TensorFlow Implementation")
    
    code = '''import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                               padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                               padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                               padding="same", use_bias=False,
                               activation="tanh")
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2),
                      padding="same", input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2),
                      padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Load dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Train the GAN
generator = build_generator()
discriminator = build_discriminator()

# ... training loop implementation'''
    
    st.code(code, language='python')
    
    st.markdown("---")
    st.markdown("### 🔑 Key Architecture Features")
    st.markdown("• **Conv2DTranspose** layers for upsampling in generator")
    st.markdown("• **BatchNormalization** for stable training")
    st.markdown("• **LeakyReLU** activation functions")
    st.markdown("• **Dropout** in discriminator to prevent overfitting")
    st.markdown("• **Tanh** activation for output layer (range: [-1, 1])")
    
    st.markdown("---")
    st.markdown("### 📦 Requirements")
    st.code("pip install streamlit tensorflow numpy matplotlib pillow", language='bash')

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with TensorFlow & Streamlit | GAN for MNIST Generation</p>
</div>
""", unsafe_allow_html=True)
