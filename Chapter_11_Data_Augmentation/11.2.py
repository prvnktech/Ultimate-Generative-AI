"""
Synthetic Data Generation App
A Streamlit application for generating synthetic images, text, and audio
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wave
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎨 Synthetic Data Generator</p>', unsafe_allow_html=True)
st.markdown("Generate synthetic images, augmented text, and audio using deep learning models")

# ============================================
# GAN Generator Class
# ============================================

class Generator(nn.Module):
    """Improved Generator with BatchNorm and LeakyReLU"""
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_text_model():
    """Load T5 model with caching"""
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model.eval()
    return tokenizer, model

def generate_images(num_images, seed=None):
    """Generate synthetic images using GAN"""
    if seed is not None:
        torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator().to(device)
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_images, 100).to(device)
        synthetic_images = generator(noise).cpu().numpy()
    
    return synthetic_images

def augment_text(texts, num_augmentations, temperature, top_k, top_p):
    """Generate augmented text using T5"""
    tokenizer, model = load_text_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Prepare inputs with translation prompt
    augmented_inputs = [f"translate English to German: {text}" for text in texts]
    
    inputs = tokenizer(
        augmented_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=30,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_augmentations,
            num_beams=1
        )
    
    augmented_texts = [
        tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]
    
    return augmented_texts

def generate_audio(duration, fundamental_freq, sample_rate, seed=None):
    """Generate synthetic audio waveform"""
    if seed is not None:
        torch.manual_seed(seed)
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Generate waveform with harmonics
    waveform = (
        0.5 * torch.sin(2 * math.pi * fundamental_freq * t) +
        0.3 * torch.sin(2 * math.pi * fundamental_freq * 2 * t) +
        0.2 * torch.sin(2 * math.pi * fundamental_freq * 3 * t) +
        0.1 * torch.sin(2 * math.pi * fundamental_freq * 4 * t) +
        0.02 * torch.randn(t.shape)
    )
    
    # Apply fade in/out
    fade_samples = int(sample_rate * 0.01)
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)
    
    waveform[:fade_samples] *= fade_in
    waveform[-fade_samples:] *= fade_out
    
    return waveform.unsqueeze(0), t

def save_wav_to_buffer(waveform, sample_rate):
    """Save waveform to buffer for download"""
    audio = waveform.squeeze().cpu().numpy()
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    audio_int16 = np.int16(audio * 32767)
    
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return buffer

# ============================================
# Sidebar Configuration
# ============================================

st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Device info
device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"🖥️ Running on: **{device}**")
st.sidebar.markdown("---")

# Seed for reproducibility
use_seed = st.sidebar.checkbox("Use random seed", value=False)
seed = st.sidebar.number_input("Seed", value=42, min_value=0) if use_seed else None

# ============================================
# Tab 1: Image Generation
# ============================================

tab1, tab2, tab3 = st.tabs(["🖼️ Image Generation", "📝 Text Augmentation", "🎵 Audio Synthesis"])

with tab1:
    st.markdown('<p class="section-header">GAN-Style Image Generation</p>', unsafe_allow_html=True)
    st.markdown("Generate synthetic 28x28 grayscale images using a GAN-style neural network")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Parameters")
        num_images = st.slider("Number of images", 1, 20, 10)
        
        if st.button("🎨 Generate Images", key="gen_img"):
            with st.spinner("Generating images..."):
                images = generate_images(num_images, seed)
                st.session_state['generated_images'] = images
                st.success(f"✅ Generated {num_images} images!")
    
    with col1:
        if 'generated_images' in st.session_state:
            images = st.session_state['generated_images']
            
            # Calculate grid dimensions
            cols = 5
            rows = (len(images) + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
            axes = axes.flatten() if rows > 1 else [axes] if len(images) == 1 else axes
            
            for i in range(len(images)):
                img = images[i].reshape(28, 28)
                img = (img + 1) / 2  # Normalize to [0, 1]
                axes[i].imshow(img, cmap="gray")
                axes[i].axis("off")
                axes[i].set_title(f"Sample {i+1}", fontsize=9)
            
            # Hide unused subplots
            for i in range(len(images), len(axes)):
                axes[i].axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("👆 Click 'Generate Images' to start")

# ============================================
# Tab 2: Text Augmentation
# ============================================

with tab2:
    st.markdown('<p class="section-header">Text Augmentation with T5</p>', unsafe_allow_html=True)
    st.markdown("Augment text by translating to German using T5 model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Texts")
        text1 = st.text_input("Text 1", "The weather is sunny.")
        text2 = st.text_input("Text 2", "I love playing football.")
        
        input_texts = [t for t in [text1, text2] if t.strip()]
    
    with col2:
        st.subheader("Parameters")
        num_aug = st.slider("Augmentations per text", 1, 5, 3)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_k = st.slider("Top-k", 10, 100, 50, 10)
        top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.05)
        
        if st.button("📝 Generate Augmentations", key="gen_text"):
            if input_texts:
                with st.spinner("Generating augmented text..."):
                    augmented = augment_text(input_texts, num_aug, temperature, top_k, top_p)
                    st.session_state['augmented_texts'] = augmented
                    st.session_state['original_texts'] = input_texts
                    st.success(f"✅ Generated {len(augmented)} augmentations!")
            else:
                st.error("Please enter at least one text!")
    
    if 'augmented_texts' in st.session_state:
        st.markdown("---")
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Texts:**")
            for i, text in enumerate(st.session_state['original_texts'], 1):
                st.markdown(f"{i}. {text}")
        
        with col2:
            st.markdown("**Augmented (German) Texts:**")
            for i, text in enumerate(st.session_state['augmented_texts'], 1):
                st.markdown(f"{i}. {text}")
    else:
        st.info("👆 Enter text and click 'Generate Augmentations' to start")

# ============================================
# Tab 3: Audio Synthesis
# ============================================

with tab3:
    st.markdown('<p class="section-header">Synthetic Audio Generation</p>', unsafe_allow_html=True)
    st.markdown("Generate synthetic audio with multiple harmonics")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Parameters")
        duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0, 0.5)
        sample_rate = st.selectbox("Sample Rate (Hz)", [8000, 16000, 44100], index=1)
        fundamental_freq = st.slider("Fundamental Frequency (Hz)", 220, 880, 440, 20)
        
        if st.button("🎵 Generate Audio", key="gen_audio"):
            with st.spinner("Generating audio..."):
                waveform, t = generate_audio(duration, fundamental_freq, sample_rate, seed)
                st.session_state['waveform'] = waveform
                st.session_state['time'] = t
                st.session_state['sample_rate'] = sample_rate
                st.session_state['duration'] = duration
                st.success("✅ Audio generated!")
    
    with col1:
        if 'waveform' in st.session_state:
            # Plot waveform
            fig, ax = plt.subplots(figsize=(10, 4))
            waveform = st.session_state['waveform']
            t = st.session_state['time']
            
            # Plot first 2000 samples for clarity
            samples_to_plot = min(2000, len(waveform.squeeze()))
            time_array = t[:samples_to_plot].numpy()
            wave_array = waveform.squeeze()[:samples_to_plot].numpy()
            
            ax.plot(time_array, wave_array, linewidth=0.5, color='#667eea')
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Synthetic Audio Waveform", fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Audio player
            st.subheader("Audio Player")
            audio_buffer = save_wav_to_buffer(waveform, st.session_state['sample_rate'])
            st.audio(audio_buffer, format='audio/wav')
            
            # Download button
            st.download_button(
                label="⬇️ Download Audio",
                data=audio_buffer,
                file_name="synthetic_audio.wav",
                mime="audio/wav"
            )
            
            # Audio info
            st.info(f"""
            **Audio Information:**
            - Duration: {st.session_state['duration']} seconds
            - Sample Rate: {st.session_state['sample_rate']} Hz
            - Fundamental Frequency: {fundamental_freq} Hz
            - Channels: Mono
            """)
        else:
            st.info("👆 Click 'Generate Audio' to start")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • PyTorch • Transformers</p>
        <p>Generate synthetic data for machine learning experiments</p>
    </div>
""", unsafe_allow_html=True)