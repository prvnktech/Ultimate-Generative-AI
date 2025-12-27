"""
GAN Architecture Lab (TensorFlow)
Interactive demonstration of Generative Adversarial Network architecture
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import io

# Page Configuration
st.set_page_config(
    page_title="GAN Lab",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.tensorflow.org/tutorials/generative/dcgan',
        'About': 'Interactive GAN Architecture Demonstration using TensorFlow'
    }
)

# Clean, Modern CSS Design System
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header - Purple/Pink Gradient for Creativity/AI */
    .hero-header {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        color: white !important;
    }

    /* Card Styling */
    .info-card, .success-card, .warning-card {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid rgba(128, 128, 128, 0.1);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Theme-aware card backgrounds */
    .info-card {
        background-color: rgba(139, 92, 246, 0.05);
        border-left: 4px solid #8B5CF6;
    }

    .success-card {
        background-color: rgba(16, 185, 129, 0.05);
        border-left: 4px solid #10b981;
    }

    .warning-card {
        background-color: rgba(245, 158, 11, 0.05);
        border-left: 4px solid #f59e0b;
    }

    /* Metric Values */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #8B5CF6;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid #e5e7eb;
        text-align: center;
        opacity: 0.8;
    }

    /* Custom Button Styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE LOGIC: GAN Implementation (TensorFlow)
# -----------------------------------------------------------------------------

def build_generator(latent_dim):
    """
    Builds the Generator model as specified in Requirement.
    Generates 28x28 grayscale images from latent noise.
    """
    model = tf.keras.Sequential(name="Generator")
    
    # Input layer - latent space
    model.add(tf.keras.layers.Dense(256, input_dim=latent_dim))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    
    # Hidden layers
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    
    # Output layer - reshape to 28x28 image
    # tanh activation puts pixel values between -1 and 1
    model.add(tf.keras.layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(tf.keras.layers.Reshape((28, 28, 1)))
    
    return model

def build_discriminator(img_shape):
    """
    Builds a basic Discriminator model.
    Classifies images as Real (1) or Fake (0).
    """
    model = tf.keras.Sequential(name="Discriminator")
    
    model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

def get_model_summary_string(model):
    """Capture model summary as a string"""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎨 GAN Architecture Lab</div>
    <div class="hero-subtitle">Building Generative Adversarial Networks with TensorFlow</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🧪 Implementation of Requirement</h4>
    <p>This lab builds the architecture for a <strong>Generative Adversarial Network (GAN)</strong>. 
    It focuses on the two competing networks: the <strong>Generator</strong> (creates synthetic data) and the <strong>Discriminator</strong> (evaluates data).</p>
</div>
""", unsafe_allow_html=True)

# Tabs for components
tab_gen, tab_disc, tab_viz = st.tabs(["1. The Generator", "2. The Discriminator", "3. Architecture Visualization"])

# ---------------------
# TAB 1: GENERATOR
# ---------------------
with tab_gen:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("### 🏭 Generator Logic")
        st.markdown("""
        The Generator takes random noise (latent vector) and transforms it into an image.
        
        **Key Layers used:**
        - `Dense`: Fully connected layers to upsample the noise.
        - `BatchNormalization`: Stabilizes training.
        - `LeakyReLU`: Activation allowing small negative values.
        - `Reshape`: Converts flat vector to 28x28 image format.
        """)
        
        # Display Code
        code = '''
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    # Input layer - latent space
    model.add(tf.keras.layers.Dense(256, input_dim=latent_dim))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    
    # Upsampling
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    
    # Output layer
    model.add(tf.keras.layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(tf.keras.layers.Reshape((28, 28, 1)))
    return model
        '''
        st.code(code, language='python')

    with col2:
        st.markdown("### ⚙️ Build Generator")
        latent_dim = st.number_input("Latent Dimension Size", min_value=10, max_value=512, value=100)
        
        if st.button("🔨 Build Generator Model", use_container_width=True):
            try:
                gen_model = build_generator(latent_dim)
                st.session_state['gen_model'] = gen_model
                st.success("Generator Built Successfully!")
            except Exception as e:
                st.error(f"Error building model: {e}")

        if 'gen_model' in st.session_state:
            st.markdown("#### Model Summary")
            st.text(get_model_summary_string(st.session_state['gen_model']))
            
            st.markdown("#### Test Generation")
            if st.button("🎲 Generate Random Image (Untrained)", type="primary", use_container_width=True):
                # Generate noise
                noise = np.random.normal(0, 1, (1, latent_dim))
                # Predict
                gen_img = st.session_state['gen_model'].predict(noise)
                # Rescale from [-1, 1] to [0, 1] for plotting
                img_plot = 0.5 * gen_img + 0.5
                img_plot = img_plot.reshape(28, 28)
                
                fig = px.imshow(img_plot, color_continuous_scale='gray', title="Generated Output (Random Noise)")
                fig.update_layout(width=300, height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig)
                st.caption("Note: Since the model is untrained, this looks like random static. Training organizes this into digits.")

# ---------------------
# TAB 2: DISCRIMINATOR
# ---------------------
with tab_disc:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("### 🕵️‍♀️ Discriminator Logic")
        st.markdown("""
        The Discriminator is a standard binary classifier. It takes an image (28x28) as input and outputs a probability (0 to 1) indicating if the image is Real or Fake.
        """)
        
        code_disc = '''
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model
        '''
        st.code(code_disc, language='python')
        
    with col2:
        st.markdown("### ⚙️ Build Discriminator")
        
        if st.button("🔨 Build Discriminator Model", use_container_width=True):
            try:
                disc_model = build_discriminator((28, 28, 1))
                st.session_state['disc_model'] = disc_model
                st.success("Discriminator Built Successfully!")
            except Exception as e:
                st.error(f"Error building model: {e}")
                
        if 'disc_model' in st.session_state:
            st.markdown("#### Model Summary")
            st.text(get_model_summary_string(st.session_state['disc_model']))

# ---------------------
# TAB 3: VISUALIZATION
# ---------------------
with tab_viz:
    st.markdown("### 🧠 GAN Architecture Flow")
    
    st.markdown("""
    The GAN process involves connecting these two models:
    1. **Noise** enters the **Generator**.
    2. **Generator** creates a **Fake Image**.
    3. **Discriminator** receives both **Real Images** (from dataset) and **Fake Images**.
    4. **Discriminator** tries to guess which is which.
    5. **Loss** is calculated to improve both:
        * Generator tries to fool Discriminator.
        * Discriminator tries to catch Generator.
    """)
    
    # Simple flowchart visualization using graphviz
    st.graphviz_chart("""
    digraph GAN {
        rankdir=LR;
        node [shape=box, style=filled, fontname="Inter"];
        
        Noise [label="Latent Noise\n(Vector)", fillcolor="#E0E7FF"];
        Generator [label="Generator\n(Model)", fillcolor="#C7D2FE"];
        FakeImg [label="Fake Image\n(28x28)", fillcolor="#E0E7FF"];
        
        RealImg [label="Real Image\n(Dataset)", fillcolor="#D1FAE5"];
        
        Discriminator [label="Discriminator\n(Model)", fillcolor="#FBCFE8"];
        
        Decision [label="Real or Fake?\n(Probability)", shape=oval, fillcolor="#FCE7F3"];
        
        Noise -> Generator;
        Generator -> FakeImg;
        FakeImg -> Discriminator;
        RealImg -> Discriminator;
        Discriminator -> Decision;
    }
    """)

# Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🎨 GAN Lab</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem;">
            Interactive exploration of Deep Learning architectures.
        </p>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • TensorFlow • Plotly
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)