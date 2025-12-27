import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(page_title="Conditional GAN (TensorFlow)", layout="wide", page_icon="🎯")

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
st.markdown("# 🎯 Conditional GAN Digit Generator (TensorFlow)")
st.markdown("### Generate specific digits (0-9) with conditional GAN")

# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100
NUM_CLASSES = 10

# Generator Model
def build_generator(noise_dim, num_classes):
    noise = layers.Input(shape=(noise_dim,))
    label = layers.Input(shape=(num_classes,))

    x = layers.Concatenate()([noise, label])
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(784, activation="tanh")(x)

    return tf.keras.Model([noise, label], x, name="generator")

# Discriminator Model
def build_discriminator(input_dim, num_classes):
    data_input = layers.Input(shape=(input_dim,), name="image_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    combined = layers.Concatenate()([data_input, label_input])
    
    x = layers.Dense(512)(combined)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    logits = layers.Dense(1)(x)

    model = tf.keras.Model(
        inputs=[data_input, label_input],
        outputs=logits,
        name="discriminator"
    )

    return model

# Loss functions
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_logits):
    return loss_fn(tf.ones_like(fake_logits), fake_logits)

def discriminator_loss(real_logits, fake_logits):
    real_labels = tf.ones_like(real_logits) * 0.9  # label smoothing
    fake_labels = tf.zeros_like(fake_logits)

    real_loss = loss_fn(real_labels, real_logits)
    fake_loss = loss_fn(fake_labels, fake_logits)
    return real_loss + fake_loss

# Training step
@tf.function
def train_step(images, labels, generator, discriminator, gen_optimizer, disc_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_logits = discriminator([images, labels], training=True)
        fake_logits = discriminator([generated_images, labels], training=True)

        disc_loss = discriminator_loss(real_logits, fake_logits)

    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # Train generator with fresh noise
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noise, labels], training=True)
        fake_logits = discriminator([generated_images, labels], training=True)
        gen_loss = generator_loss(fake_logits)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    return gen_loss, disc_loss

# Generate digit function
def generate_digit(generator, digit, n=16):
    noise = tf.random.normal([n, NOISE_DIM])
    labels = tf.one_hot([digit] * n, NUM_CLASSES)

    images = generator([noise, labels], training=False)
    images = images.numpy().reshape(n, 28, 28)
    
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2

    return images

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'discriminator' not in st.session_state:
    st.session_state.discriminator = None
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'selected_digit' not in st.session_state:
    st.session_state.selected_digit = 0

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎨 Generate", "🏋️ Train", "💻 Code"])

# Tab 1: Generate
with tab1:
    st.markdown("### 🎨 Generate Specific Digits")
    
    if st.session_state.generator is None:
        st.info("👆 No trained model available. Please train the model first in the 'Train' tab.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Select Digit (0-9)")
            
            cols = st.columns(5)
            selected_digit = None
            
            for i in range(10):
                col_idx = i % 5
                with cols[col_idx]:
                    if st.button(str(i), key=f"digit_{i}", use_container_width=True):
                        selected_digit = i
            
            if selected_digit is not None:
                st.session_state.selected_digit = selected_digit
            
            st.markdown(f"**Selected Digit:** {st.session_state.selected_digit}")
            
            num_samples = st.slider("Number of samples", 4, 25, 16)
            
            if st.button("✨ Generate Digit", type="primary"):
                with st.spinner("Generating..."):
                    images = generate_digit(
                        st.session_state.generator,
                        st.session_state.selected_digit,
                        num_samples
                    )
                    st.session_state.generated_images = images
                
                st.success(f"✅ Generated {num_samples} samples of digit {st.session_state.selected_digit}!")
        
        with col2:
            st.markdown("#### Generated Output")
            
            if 'generated_images' in st.session_state:
                images = st.session_state.generated_images
                
                # Create matplotlib figure
                grid_size = int(np.ceil(np.sqrt(len(images))))
                fig = plt.figure(figsize=(10, 10))
                fig.suptitle(f"Generated Digit: {st.session_state.selected_digit}", 
                           fontsize=16, fontweight='bold')
                
                for i in range(len(images)):
                    plt.subplot(grid_size, grid_size, i + 1)
                    plt.imshow(images[i], cmap="gray")
                    plt.axis("off")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("👆 Click 'Generate Digit' to create images")
        
        st.markdown("---")
        st.markdown("#### 🎲 Quick Actions")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🎲 Generate Random Digit"):
                random_digit = np.random.randint(0, 10)
                st.session_state.selected_digit = random_digit
                
                with st.spinner("Generating..."):
                    images = generate_digit(
                        st.session_state.generator,
                        random_digit,
                        16
                    )
                    st.session_state.generated_images = images
                
                st.success(f"✅ Generated random digit: {random_digit}")
                st.rerun()
        
        with col_b:
            if st.button("🔟 Generate All Digits (0-9)"):
                with st.spinner("Generating all digits..."):
                    fig = plt.figure(figsize=(15, 6))
                    
                    for digit in range(10):
                        images = generate_digit(st.session_state.generator, digit, 4)
                        
                        for i in range(4):
                            plt.subplot(4, 10, digit + 1 + i * 10)
                            plt.imshow(images[i], cmap="gray")
                            plt.axis("off")
                            if i == 0:
                                plt.title(f"Digit {digit}", fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# Tab 2: Train
with tab2:
    st.markdown("### 🏋️ Train Conditional GAN Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Training Configuration")
        
        epochs = st.slider("Number of Epochs", 1, 100, 20)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}"
        )
        
        if st.button("🚀 Start Training", type="primary"):
            st.session_state.training_started = True
            
            with st.spinner("Loading dataset..."):
                # Load and prepare dataset
                (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
                
                x_train = x_train.astype("float32") / 127.5 - 1.0
                x_train = x_train.reshape(-1, 784)
                
                y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
                
                dataset = (
                    tf.data.Dataset.from_tensor_slices((x_train, y_train))
                    .shuffle(BUFFER_SIZE)
                    .batch(BATCH_SIZE, drop_remainder=True)
                )
            
            # Initialize models
            generator = build_generator(NOISE_DIM, NUM_CLASSES)
            discriminator = build_discriminator(784, NUM_CLASSES)
            
            gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            
            # Progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_metrics = st.empty()
            preview_placeholder = st.empty()
            
            # Training loop
            for epoch in range(epochs):
                status_text.text(f"Training Epoch {epoch + 1}/{epochs}...")
                
                for image_batch, label_batch in dataset:
                    g_loss, d_loss = train_step(
                        image_batch, label_batch,
                        generator, discriminator,
                        gen_optimizer, disc_optimizer
                    )
                
                # Update progress
                progress_bar.progress((epoch + 1) / epochs)
                epoch_metrics.metric(
                    f"Epoch {epoch + 1}",
                    f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}"
                )
                
                # Show preview every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    with preview_placeholder.container():
                        st.markdown("##### Training Progress Preview")
                        fig = plt.figure(figsize=(10, 4))
                        
                        for digit in range(10):
                            images = generate_digit(generator, digit, 1)
                            plt.subplot(2, 5, digit + 1)
                            plt.imshow(images[0], cmap="gray")
                            plt.title(f"Digit {digit}", fontsize=10)
                            plt.axis("off")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            
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
        else:
            st.info("⏳ Training not started")
        
        st.markdown("---")
        st.markdown("#### About Conditional GAN")
        st.markdown("""
        This conditional GAN allows you to:
        - **Control** which digit to generate (0-9)
        - **Condition** the generation on class labels
        - **Train** with label information for better results
        
        **Architecture:**
        - Generator takes noise + one-hot label
        - Discriminator takes image + one-hot label
        - Uses label smoothing for stability
        - Dense layers with BatchNorm and LeakyReLU
        """)
        
        st.markdown("---")
        st.markdown("#### Training Tips")
        st.markdown("""
        - Start with 20-30 epochs for good results
        - Use learning rate of 2e-4 (default)
        - Label smoothing helps stability
        - Higher epochs = better quality
        """)

# Tab 3: Code
with tab3:
    st.markdown("### 💻 TensorFlow Conditional GAN Implementation")
    
    code = '''import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Constants
BATCH_SIZE = 256
NOISE_DIM = 100
NUM_CLASSES = 10

# Conditional Generator
def build_generator(noise_dim, num_classes):
    noise = layers.Input(shape=(noise_dim,))
    label = layers.Input(shape=(num_classes,))
    
    x = layers.Concatenate()([noise, label])
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(784, activation="tanh")(x)
    
    return tf.keras.Model([noise, label], x, name="generator")

# Conditional Discriminator
def build_discriminator(input_dim, num_classes):
    data_input = layers.Input(shape=(input_dim,))
    label_input = layers.Input(shape=(num_classes,))
    
    combined = layers.Concatenate()([data_input, label_input])
    
    x = layers.Dense(512)(combined)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    logits = layers.Dense(1)(x)
    
    return tf.keras.Model([data_input, label_input], logits)

# Generate specific digit
def generate_digit(generator, digit, n=16):
    noise = tf.random.normal([n, NOISE_DIM])
    labels = tf.one_hot([digit] * n, NUM_CLASSES)
    
    images = generator([noise, labels], training=False)
    images = images.numpy().reshape(n, 28, 28)
    images = (images + 1) / 2  # Denormalize
    
    return images

# Initialize and use
generator = build_generator(NOISE_DIM, NUM_CLASSES)
discriminator = build_discriminator(784, NUM_CLASSES)

# Generate digit 5
generated_fives = generate_digit(generator, 5, 16)'''
    
    st.code(code, language='python')
    
    st.markdown("---")
    st.markdown("### 🔑 Key Features")
    st.markdown("• **Conditional Generation** - Control which digit to generate")
    st.markdown("• **One-hot Encoding** - Labels concatenated with noise/images")
    st.markdown("• **Label Smoothing** - Uses 0.9 instead of 1.0 for stability")
    st.markdown("• **BatchNormalization** - Stabilizes training")
    st.markdown("• **LeakyReLU(0.2)** - Prevents dying neurons")
    st.markdown("• **Dropout(0.3)** - Prevents discriminator overfitting")
    
    st.markdown("---")
    st.markdown("### 📦 Requirements")
    st.code("pip install streamlit tensorflow numpy matplotlib pillow", language='bash')
    
    st.markdown("---")
    st.markdown("### 🎯 Use Cases")
    st.markdown("""
    Conditional GANs can be used for:
    - **Targeted generation** of specific classes
    - **Data augmentation** for specific categories
    - **Style transfer** with class control
    - **Text-to-image** generation (with text embeddings)
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with TensorFlow & Streamlit | Conditional GAN for MNIST</p>
</div>
""", unsafe_allow_html=True)
