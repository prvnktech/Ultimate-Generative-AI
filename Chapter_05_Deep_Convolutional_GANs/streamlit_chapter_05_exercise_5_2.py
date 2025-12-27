import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="GAN MNIST Generator (PyTorch)", layout="wide", page_icon="⚡")

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
st.markdown("# ⚡ GAN MNIST Generator (PyTorch)")
st.markdown("### Generate handwritten digits using PyTorch-based GAN")

# Constants
BATCH_SIZE = 256
NOISE_DIM = 100
LR = 1e-4
BETA1 = 0.5
IMAGE_SIZE = 28
OUTPUT_DIR = "generated_images_pt"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 7 * 7 * 256, bias=False),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(256, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 1, 5, 2, 2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.net(x)

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
            st.info("👆 No trained model available. Please train the model first in the 'Train' tab or load existing images from the 'generated_images_pt' folder.")
        
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
                    st.session_state.generator.eval()
                    with torch.no_grad():
                        noise = torch.randn(num_images, NOISE_DIM, device=device)
                        generated_images = st.session_state.generator(noise).cpu()
                    
                    # Denormalize from [-1,1] to [0,1]
                    generated_images = (generated_images + 1) / 2
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
                    plt.imshow(generated_images[i][0], cmap="gray")
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
        
        st.markdown(f"**Device:** {device.type.upper()}")
        if torch.cuda.is_available():
            st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.info("💻 Using CPU")
        
        if st.button("🚀 Start Training", type="primary"):
            st.session_state.training_started = True
            
            with st.spinner("Loading dataset..."):
                # Load and prepare dataset
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                
                dataset = datasets.MNIST(
                    root="./data",
                    train=True,
                    download=True,
                    transform=transform
                )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True
                )
            
            # Initialize models
            G = Generator().to(device)
            D = Discriminator().to(device)
            
            # Loss and optimizers
            criterion = nn.BCEWithLogitsLoss()
            optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(BETA1, 0.999))
            optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(BETA1, 0.999))
            
            # Fixed noise for visualization
            fixed_noise = torch.randn(16, NOISE_DIM, device=device)
            
            # Progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_metrics = st.empty()
            
            # Training loop
            for epoch in range(1, epochs + 1):
                status_text.text(f"Training Epoch {epoch}/{epochs}...")
                
                for real_images, _ in dataloader:
                    real_images = real_images.to(device)
                    batch_size = real_images.size(0)
                    
                    # Train Discriminator
                    noise = torch.randn(batch_size, NOISE_DIM, device=device)
                    fake_images = G(noise)
                    
                    real_labels = torch.ones(batch_size, 1, device=device)
                    fake_labels = torch.zeros(batch_size, 1, device=device)
                    
                    D_real = D(real_images)
                    D_fake = D(fake_images.detach())
                    
                    loss_D_real = criterion(D_real, real_labels)
                    loss_D_fake = criterion(D_fake, fake_labels)
                    loss_D = loss_D_real + loss_D_fake
                    
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    
                    # Train Generator
                    output = D(fake_images)
                    loss_G = criterion(output, real_labels)
                    
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
                
                # Save images every 5 epochs or last epoch
                if epoch % 5 == 0 or epoch == epochs:
                    G.eval()
                    with torch.no_grad():
                        fake_images = G(fixed_noise).cpu()
                    G.train()
                    
                    fake_images = (fake_images + 1) / 2  # [-1,1] → [0,1]
                    
                    fig = plt.figure(figsize=(4, 4))
                    for i in range(16):
                        plt.subplot(4, 4, i + 1)
                        plt.imshow(fake_images[i][0], cmap="gray")
                        plt.axis("off")
                    
                    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch_{epoch:04d}.png"))
                    plt.close(fig)
                
                # Update progress
                progress_bar.progress(epoch / epochs)
                epoch_metrics.metric(
                    f"Epoch {epoch}",
                    f"G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f}"
                )
            
            # Save models to session state
            st.session_state.generator = G
            st.session_state.discriminator = D
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
        - **Generator**: ConvTranspose2d layers for upsampling
        - **Discriminator**: Conv2d layers for classification
        - **Training**: Binary cross-entropy with logits loss
        - **Optimizer**: Adam with beta1=0.5
        - **Dataset**: MNIST handwritten digits
        """)

# Tab 3: Code
with tab3:
    st.markdown("### 💻 PyTorch Implementation")
    
    code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Constants
BATCH_SIZE = 256
NOISE_DIM = 100
LR = 1e-4
BETA1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 7 * 7 * 256, bias=False),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            
            nn.ConvTranspose2d(256, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 1, 5, 2, 2, output_padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

# ... training loop implementation'''
    
    st.code(code, language='python')
    
    st.markdown("---")
    st.markdown("### 🔑 Key Architecture Features")
    st.markdown("• **ConvTranspose2d** layers for upsampling in generator")
    st.markdown("• **BatchNorm1d/2d** for stable training")
    st.markdown("• **LeakyReLU(0.2)** activation functions")
    st.markdown("• **Dropout(0.3)** in discriminator to prevent overfitting")
    st.markdown("• **Tanh** activation for output layer (range: [-1, 1])")
    st.markdown("• **BCEWithLogitsLoss** for training stability")
    
    st.markdown("---")
    st.markdown("### 📦 Requirements")
    st.code("pip install streamlit torch torchvision numpy matplotlib pillow", language='bash')

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with PyTorch & Streamlit | GAN for MNIST Generation</p>
</div>
""", unsafe_allow_html=True)
