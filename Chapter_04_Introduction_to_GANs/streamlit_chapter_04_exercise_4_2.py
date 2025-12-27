"""
GAN Architecture Lab (PyTorch)
Interactive demonstration of Generative Adversarial Network architecture using PyTorch
Includes Optimizers, Training Loop Logic, and Visualization
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import io

# Page Configuration
st.set_page_config(
    page_title="PyTorch GAN Lab",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html',
        'About': 'Interactive GAN Architecture Demonstration using PyTorch'
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

    /* Hero Header - Orange/Red Gradient for PyTorch */
    .hero-header {
        background: linear-gradient(135deg, #EE4C2C 0%, #E65C00 100%);
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
    .info-card {
        background-color: rgba(238, 76, 44, 0.05);
        border-left: 4px solid #EE4C2C;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE LOGIC: GAN Implementation (PyTorch)
# -----------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh() # Output ranges from -1 to 1
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

def visualize_images_matplotlib(generator, latent_dim):
    """
    Visualizes generated images using Matplotlib as requested.
    Returns the figure object to be displayed in Streamlit.
    """
    # Generate noise
    z = torch.randn(25, latent_dim)
    
    # Generate images (no gradient needed for visualization)
    with torch.no_grad():
        gen_imgs = generator(z).detach().cpu()
    
    # Rescale images 0-1
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔥 PyTorch GAN Lab</div>
    <div class="hero-subtitle">Generative Adversarial Networks with Stochastic Gradient Descent</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🧪 Requirement 4.2 Implementation</h4>
    <p>This lab translates the GAN architecture into <strong>PyTorch</strong>. It builds upon the previous requirement by adding <strong>Optimizers (Adam)</strong> and detailing the <strong>Training Loop</strong> logic.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
# Correctly defining tabs based on usage below
tab_arch, tab_train, tab_viz = st.tabs(["1. Generator & Discriminator", "2. Optimizers & Training", "3. Visualization"])

# ---------------------
# TAB 1: ARCHITECTURE
# ---------------------
with tab_arch:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏭 Generator (PyTorch)")
        st.markdown("Uses `nn.Linear`, `nn.BatchNorm1d`, and `nn.LeakyReLU`.")
        st.code('''
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # ... more layers ...
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img
        ''', language='python')
        
    with col2:
        st.markdown("### 🕵️‍♀️ Discriminator (PyTorch)")
        st.markdown("A simple classifier outputting a validity score (0-1).")
        st.code('''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
        ''', language='python')

    st.divider()
    
    # Interactive Builder
    st.subheader("🛠️ Build Models")
    latent_dim = st.slider("Latent Dimension", 10, 200, 100)
    
    if st.button("Initialize PyTorch Models"):
        gen = Generator(latent_dim)
        disc = Discriminator()
        st.session_state['pt_gen'] = gen
        st.session_state['pt_disc'] = disc
        st.success("Models initialized in PyTorch!")
        
        with st.expander("View Model Architecture"):
            st.write("Generator:", gen)
            st.write("Discriminator:", disc)

# ---------------------
# TAB 2: OPTIMIZERS & TRAINING
# ---------------------
with tab_train:
    st.markdown("### ⚡ Optimizers (Adam)")
    st.markdown("We use **Adam** (Adaptive Moment Estimation) as it is generally preferred for GANs due to its adaptability.")
    
    st.code('''
# Optimizers for generator and discriminator
# Beta1 is often set to 0.5 for stable GAN training
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    ''', language='python')
    
    st.markdown("### 🔄 Training Loop Logic")
    st.markdown("In PyTorch, we manually manage the gradients. The loop consists of two main phases per batch:")
    
    st.code('''
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        real_loss = adversarial_loss(discriminator(imgs), valid)
        
        # Fake images
        z = torch.randn(imgs.shape[0], latent_dim)
        gen_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        
        # Total Discriminator Loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Generator wants Discriminator to label fake images as valid
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        g_loss.backward()
        optimizer_G.step()
    ''', language='python')

# ---------------------
# TAB 3: VISUALIZATION
# ---------------------
with tab_viz:
    st.markdown("### 🖼️ Visualizing Progress")
    st.markdown("Using `matplotlib`, we generate a grid of images from random noise to inspect the Generator's progress.")
    
    if 'pt_gen' not in st.session_state:
        st.warning("Please initialize models in Tab 1 first.")
    else:
        if st.button("Generate & Visualize Batch"):
            gen_model = st.session_state['pt_gen']
            
            # Switch to eval mode
            gen_model.eval()
            
            # Create figure
            fig = visualize_images_matplotlib(gen_model, latent_dim)
            
            # Display
            st.pyplot(fig)
            st.caption("Since the model is untrained, these images represent the initial random weights mapped to pixel space.")
            
            # Show the code used for this
            st.markdown("#### Matplotlib Visualization Code")
            st.code('''
def visualize_images(epoch, generator, latent_dim):
    noise = np.random.normal(0, 1, (25, latent_dim))
    # PyTorch specific: Convert to tensor
    noise_tensor = torch.FloatTensor(noise)
    gen_images = generator(noise_tensor)
    gen_images = 0.5 * gen_images + 0.5  # Rescale images 0-1

    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        # Reshape for matplotlib (H, W) or (H, W, C)
        plt.imshow(gen_images[i, 0, :, :].detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
            ''', language='python')

# Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🔥 PyTorch GAN Lab</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem;">
            Deep Learning Architecture & Optimization.
        </p>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • PyTorch • Matplotlib
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)