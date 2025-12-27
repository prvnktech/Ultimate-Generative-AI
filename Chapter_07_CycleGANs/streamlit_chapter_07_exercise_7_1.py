"""
CycleGAN Architecture Lab (PyTorch)
Interactive demonstration of CycleGAN architecture components and training logic.
Focuses on unpaired image-to-image translation.
"""

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io

# Page Configuration
st.set_page_config(
    page_title="CycleGAN Lab",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://junyanz.github.io/CycleGAN/',
        'About': 'Interactive CycleGAN Architecture Demonstration using PyTorch'
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

    /* Hero Header - Teal/Blue Gradient for CycleGAN */
    .hero-header {
        background: linear-gradient(135deg, #0D9488 0%, #2563EB 100%);
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
        background-color: rgba(13, 148, 136, 0.05);
        border-left: 4px solid #0D9488;
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
# CORE LOGIC: CycleGAN Implementation (PyTorch)
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

def generator_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, n_res_blocks=9):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 1, 0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            generator_block(64, 128, 3, 2, 1),
            generator_block(128, 256, 3, 2, 1),

            # Residual blocks
            *[ResidualBlock(256) for _ in range(n_res_blocks)],

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

def plot_images(real_A, fake_B, reconstructed_A):
    """
    Visualizes the translation cycle: Real A -> Fake B -> Reconstructed A
    """
    # Detach and convert to numpy, move channels to last dim for matplotlib
    # Assuming standard normalization (mean=0.5, std=0.5) -> (img * 0.5 + 0.5)
    def prep_img(tensor):
        img = tensor.detach().cpu().squeeze(0)
        img = img.permute(1, 2, 0).numpy()
        img = 0.5 * img + 0.5
        return np.clip(img, 0, 1)

    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(prep_img(real_A))
    plt.title("Real Domain A")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(prep_img(fake_B))
    plt.title("Generated Domain B (Fake)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prep_img(reconstructed_A))
    plt.title("Reconstructed Domain A")
    plt.axis("off")
    
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔄 CycleGAN Architecture Lab</div>
    <div class="hero-subtitle">Unpaired Image-to-Image Translation with Cycle Consistency</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🧪 Requirement 7.1 Implementation</h4>
    <p>This lab demonstrates the <strong>CycleGAN</strong> architecture using PyTorch. 
    Unlike standard GANs, CycleGAN learns to translate between two domains (e.g., Horse ↔ Zebra) without paired training data by enforcing <strong>Cycle Consistency</strong>.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_arch, tab_train, tab_viz = st.tabs(["1. Architecture Components", "2. CycleGAN Training Logic", "3. Visualization"])

# ---------------------
# TAB 1: ARCHITECTURE
# ---------------------
with tab_arch:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏭 Generator (ResNet-based)")
        st.markdown("Uses Residual Blocks to maintain content structure while changing style.")
        st.code('''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Encoding (Downsampling)
            generator_block(3, 64, 7, 1, 3),
            generator_block(64, 128, 3, 2, 1),
            generator_block(128, 256, 3, 2, 1),
            
            # Transformation (Residual Blocks)
            *[ResidualBlock(256) for _ in range(9)],
            
            # Decoding (Upsampling)
            nn.ConvTranspose2d(256, 128, ...),
            nn.ConvTranspose2d(128, 64, ...),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
        ''', language='python')
        
    with col2:
        st.markdown("### 🧱 Residual Block")
        st.markdown("The core component allowing deep networks without vanishing gradients.")
        st.code('''
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)
        ''', language='python')

    st.divider()
    
    # Interactive Builder
    st.subheader("🛠️ Build CycleGAN Components")
    
    if st.button("Initialize PyTorch Models"):
        gen_AB = Generator() # Domain A -> B
        gen_BA = Generator() # Domain B -> A
        disc_A = Discriminator()
        disc_B = Discriminator()
        
        st.session_state['cycle_models'] = {
            'G_AB': gen_AB,
            'G_BA': gen_BA
        }
        st.success("Initialized 2 Generators (A→B, B→A) and 2 Discriminators!")
        
        with st.expander("View Generator Architecture"):
            st.write(gen_AB)

# ---------------------
# TAB 2: TRAINING LOGIC
# ---------------------
with tab_train:
    st.markdown("### 🔄 The Cycle Consistency Loss")
    st.markdown("""
    The magic of CycleGAN is the **Cycle Consistency Loss**.
    If we translate an image from A to B, and then back from B to A, we should get the original image back.
    
    $$ A \\rightarrow G_{AB}(A) \\rightarrow G_{BA}(G_{AB}(A)) \\approx A $$
    """)
    
    st.code('''
# Training Loop Pseudocode
for i, (imgs_A, imgs_B) in enumerate(dataloader):
    
    # ------ Train Generators ------
    optimizer_G.zero_grad()
    
    # 1. GAN Loss (Fooling Discriminators)
    fake_B = G_AB(real_A)
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    
    fake_A = G_BA(real_B)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    
    # 2. Cycle Consistency Loss (Reconstruction)
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)
    
    # Total Generator Loss
    loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * (loss_cycle_A + loss_cycle_B)
    loss_G.backward()
    optimizer_G.step()
    
    # ------ Train Discriminators ------
    # ... (Train D_A and D_B on real vs fake images)
    ''', language='python')

# ---------------------
# TAB 3: VISUALIZATION
# ---------------------
with tab_viz:
    st.markdown("### 🖼️ Translation Visualization")
    st.markdown("Visualize the translation cycle: **Real A → Fake B → Reconstructed A**")
    
    if 'cycle_models' not in st.session_state:
        st.warning("Please initialize models in Tab 1 first.")
    else:
        if st.button("Simulate Translation Cycle"):
            models = st.session_state['cycle_models']
            G_AB = models['G_AB']
            G_BA = models['G_BA']
            
            # Switch to eval mode
            G_AB.eval()
            G_BA.eval()
            
            # Create a "dummy" Real Image A (Random Noise mimicking an image)
            # In a real app, this would be a loaded photo
            real_A = torch.randn(1, 3, 256, 256)
            
            with torch.no_grad():
                # Forward pass: A -> B
                fake_B = G_AB(real_A)
                # Backward pass: B -> A (Reconstruction)
                recov_A = G_BA(fake_B)
            
            # Plot
            fig = plot_images(real_A, fake_B, recov_A)
            st.pyplot(fig)
            
            st.caption("Note: Since models are untrained, the output is random noise. "
                       "In a trained model, 'Real A' might be a Horse, 'Fake B' a Zebra, "
                       "and 'Reconstructed A' the original Horse.")

# Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🔄 CycleGAN Lab</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem;">
            Advanced Unpaired Image-to-Image Translation.
        </p>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • PyTorch • Matplotlib
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)