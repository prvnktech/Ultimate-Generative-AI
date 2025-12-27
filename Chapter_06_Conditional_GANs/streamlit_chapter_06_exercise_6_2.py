import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="cGAN Digit Generator", layout="wide", page_icon="⚡")

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
st.markdown("# ⚡ Conditional GAN Digit Generator")
st.markdown("### PyTorch-powered digit generation from trained model")

# Constants
NOISE_DIM = 100
NUM_CLASSES = 10

# Conditional Generator (PyTorch) - Matching your notebook exactly
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.net(x)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'selected_digit' not in st.session_state:
    st.session_state.selected_digit = 0

# Create tabs
tab1, tab2 = st.tabs(["🎨 Generate", "💻 Code"])

# Generate Tab
with tab1:
    # Try to load model if not already loaded
    if not st.session_state.model_loaded:
        try:
            with st.spinner("Loading trained model..."):
                # Initialize generator
                generator = Generator(NOISE_DIM, NUM_CLASSES)
                
                # Load checkpoint
                checkpoint = torch.load('gan_checkpoint.pth', map_location=torch.device('cpu'))
                
                # Load generator state dict (adjust key if needed)
                if 'generator' in checkpoint:
                    generator.load_state_dict(checkpoint['generator'])
                elif 'generator_state_dict' in checkpoint:
                    generator.load_state_dict(checkpoint['generator_state_dict'])
                else:
                    # Assume the checkpoint is the state dict itself
                    generator.load_state_dict(checkpoint)
                
                generator.eval()
                st.session_state.generator = generator
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Model file 'gan_checkpoint.pth' not found. Please ensure the file is in the same directory.")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    if st.session_state.model_loaded:
        st.markdown("### 🎨 Generate Digit")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Select digit to generate (0-9)")
            
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
            
            if st.button("✨ Generate Digit", type="primary"):
                with st.spinner("Generating..."):
                    # Generate 6 samples of the digit using the trained generator
                    n_samples = 6
                    noise = torch.randn(n_samples, NOISE_DIM)
                    label = torch.zeros(n_samples, NUM_CLASSES)
                    label[:, st.session_state.selected_digit] = 1
                    
                    with torch.no_grad():
                        generated = st.session_state.generator(noise, label)
                        generated_images = generated.view(n_samples, 28, 28).cpu().numpy()
                        
                        # Denormalize from [-1, 1] to [0, 1]
                        generated_images = (generated_images + 1) / 2
                        
                        st.session_state.generated_images = generated_images
                
                st.success(f"✅ Generated 6 samples of digit {st.session_state.selected_digit}!")
        
        with col2:
            st.markdown("#### Generated Output")
            
            if 'generated_images' in st.session_state:
                fig, axes = plt.subplots(2, 3, figsize=(8, 6))
                fig.suptitle(f"Generated Digit: {st.session_state.selected_digit}", 
                           fontsize=16, fontweight='bold')
                
                for idx, ax in enumerate(axes.flat):
                    ax.imshow(st.session_state.generated_images[idx], cmap='gray')
                    ax.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("👆 Click 'Generate Digit' to create images")
            
            st.markdown("---")
            st.markdown("**How it works:**")
            st.markdown("• Generator receives random noise + class label")
            st.markdown("• Pre-trained on MNIST dataset")
            st.markdown("• Each generation is unique due to random noise")
            
            if st.button("🎲 Generate Random Digit"):
                random_digit = np.random.randint(0, 10)
                st.session_state.selected_digit = random_digit
                
                with st.spinner("Generating..."):
                    n_samples = 6
                    noise = torch.randn(n_samples, NOISE_DIM)
                    label = torch.zeros(n_samples, NUM_CLASSES)
                    label[:, random_digit] = 1
                    
                    with torch.no_grad():
                        generated = st.session_state.generator(noise, label)
                        generated_images = generated.view(n_samples, 28, 28).cpu().numpy()
                        generated_images = (generated_images + 1) / 2
                        
                        st.session_state.generated_images = generated_images
                
                st.success(f"✅ Generated 6 samples of random digit {random_digit}!")
                st.rerun()

# Code Tab
with tab2:
    st.markdown("### 💻 PyTorch Implementation")
    
    code = '''import torch
import torch.nn as nn

# Constants
NOISE_DIM = 100
NUM_CLASSES = 10

# Conditional Generator (matches your notebook)
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.net(x)

# Load trained model
generator = Generator(NOISE_DIM, NUM_CLASSES)
checkpoint = torch.load('gan_checkpoint.pth', map_location='cpu')
generator.load_state_dict(checkpoint['generator'])
generator.eval()

# Generate a digit
with torch.no_grad():
    noise = torch.randn(1, NOISE_DIM)
    label = torch.zeros(1, NUM_CLASSES)
    label[0, 5] = 1  # Generate digit 5
    
    generated = generator(noise, label)
    generated_image = generated.view(28, 28)'''
    
    st.code(code, language='python')
    
    st.markdown("---")
    st.markdown("### 🔑 Key Architecture Features")
    st.markdown("• **BatchNormalization** for stable training")
    st.markdown("• **LeakyReLU(0.2)** activation functions")
    st.markdown("• **Tanh** activation for pixel values [-1, 1]")
    st.markdown("• **Conditional generation** with one-hot encoded labels")
    st.markdown("• **Architecture**: 110 → 256 → 512 → 1024 → 784")
    
    st.markdown("---")
    st.markdown("### 📦 Requirements")
    st.code("pip install streamlit torch numpy matplotlib pillow", language='bash')
    
    st.markdown("---")
    st.markdown("### 📁 Checkpoint Structure")
    st.markdown("""
    The code expects `gan_checkpoint.pth` to contain:
    ```python
    {
        'generator': generator.state_dict(),
        # ... other keys (optional)
    }
    ```
    
    Alternative formats supported:
    - `{'generator_state_dict': ...}`
    - Direct state_dict (no wrapper dict)
    """)