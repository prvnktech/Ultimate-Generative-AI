import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io

st.set_page_config(page_title="Diffusion Model Tutorial", layout="wide")

# Title and Introduction
st.title("🎨 Interactive Diffusion Model Tutorial")
st.markdown("""
This app demonstrates a **minimal diffusion model** that generates simple 2D shapes.
Learn how diffusion models add and remove noise to create new data!
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
num_steps = st.sidebar.slider("Diffusion Steps", 10, 100, 30)
shape_type = st.sidebar.selectbox("Shape Type", ["circle", "square", "triangle"])

# Create shapes
@st.cache_data
def create_2d_shapes():
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)]).T * 0.5
    square = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    triangle = np.array([[0, 0.6], [-0.5, -0.3], [0.5, -0.3], [0, 0.6]])
    return circle, square, triangle

circle, square, triangle = create_2d_shapes()
shapes = {'circle': circle, 'square': square, 'triangle': triangle}

# Forward Diffusion Class
class ForwardDiffusion:
    def __init__(self, steps=50):
        self.steps = steps
        self.betas = np.linspace(0.01, 0.3, steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
    
    def add_noise(self, x, t):
        noise = np.random.randn(*x.shape)
        sqrt_alpha_bar = np.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bars[t])
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        return noisy_x, noise

# Simple Denoiser
class SimpleDenoiser:
    def __init__(self, forward_process):
        self.forward = forward_process
        
    def denoise_step(self, x_t, t):
        if t == 0:
            return x_t
        denoised = x_t * 0.9 + np.random.randn(*x_t.shape) * 0.1
        return denoised

# Generate shape function
def generate_shape(forward, denoiser, target_shape):
    noise = np.random.randn(*target_shape.shape)
    reverse_steps = []
    current = noise.copy()
    
    for t in reversed(range(forward.steps)):
        current = denoiser.denoise_step(current, t)
        if t % (forward.steps // 4) == 0 or t < 5:
            reverse_steps.append((t, current.copy()))
    
    return noise, current, reverse_steps

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Original Shapes", "➕ Forward Process", "➖ Reverse Process", "🎯 Generate"])

# Tab 1: Show original shapes
with tab1:
    st.header("Original 2D Shapes")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].scatter(circle[:, 0], circle[:, 1], s=20, alpha=0.7, c='blue')
    axes[0].set_title('Circle', fontsize=14)
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(square[:, 0], square[:, 1], s=20, alpha=0.7, c='green')
    axes[1].set_title('Square', fontsize=14)
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(triangle[:, 0], triangle[:, 1], s=20, alpha=0.7, c='red')
    axes[2].set_title('Triangle', fontsize=14)
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tab 2: Forward Process
with tab2:
    st.header("Forward Diffusion: Adding Noise")
    st.markdown("Watch how the shape gradually becomes pure noise!")
    
    forward = ForwardDiffusion(steps=num_steps)
    selected_shape = shapes[shape_type]
    
    # Show key steps
    key_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
    
    fig, axes = plt.subplots(1, len(key_steps), figsize=(15, 3))
    for i, step_idx in enumerate(key_steps):
        if step_idx == 0:
            step = selected_shape
        else:
            step, _ = forward.add_noise(selected_shape, step_idx)
        
        axes[i].scatter(step[:, 0], step[:, 1], s=15, alpha=0.7)
        axes[i].set_title(f'Step {step_idx}', fontsize=12)
        axes[i].set_xlim(-2, 2)
        axes[i].set_ylim(-2, 2)
        axes[i].grid(True, alpha=0.3)
        
        if step_idx > 0:
            noise_level = 1 - forward.alpha_bars[step_idx]
            axes[i].text(0.5, -0.15, f'Noise: {noise_level:.2f}', 
                        transform=axes[i].transAxes, ha='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.info(f"💡 The {shape_type} becomes increasingly noisy over {num_steps} steps!")

# Tab 3: Reverse Process Visualization
with tab3:
    st.header("Reverse Diffusion: Removing Noise")
    st.markdown("Start from pure noise and gradually reveal structure!")
    
    forward = ForwardDiffusion(steps=num_steps)
    denoiser = SimpleDenoiser(forward)
    selected_shape = shapes[shape_type]
    
    noise_start, generated_shape, steps_reverse = generate_shape(forward, denoiser, selected_shape)
    
    # Show reverse process
    num_reverse_steps = min(5, len(steps_reverse))
    fig, axes = plt.subplots(1, num_reverse_steps + 1, figsize=(16, 3))
    
    # Pure noise
    axes[0].scatter(noise_start[:, 0], noise_start[:, 1], s=15, alpha=0.7)
    axes[0].set_title('Pure Noise', fontsize=12)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].grid(True, alpha=0.3)
    
    # Intermediate steps
    for i in range(num_reverse_steps):
        if i < len(steps_reverse):
            t, points = steps_reverse[i]
            axes[i+1].scatter(points[:, 0], points[:, 1], s=15, alpha=0.7)
            axes[i+1].set_title(f'Step {num_steps-t}', fontsize=12)
        else:
            axes[i+1].scatter(generated_shape[:, 0], generated_shape[:, 1], s=15, alpha=0.7)
            axes[i+1].set_title('Final', fontsize=12)
        
        axes[i+1].set_xlim(-2, 2)
        axes[i+1].set_ylim(-2, 2)
        axes[i+1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.success("✨ Structure emerges from noise! Real models use neural networks to do this much better.")

# Tab 4: Interactive Generation
with tab4:
    st.header("🎯 Generate New Shapes")
    st.markdown("Click the button to generate a new shape from random noise!")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Controls")
        if st.button("🎲 Generate New Shape", type="primary"):
            st.session_state.generate_new = True
        
        show_comparison = st.checkbox("Show Original", value=True)
    
    with col2:
        if 'generate_new' not in st.session_state:
            st.session_state.generate_new = True
        
        if st.session_state.generate_new:
            forward = ForwardDiffusion(steps=num_steps)
            denoiser = SimpleDenoiser(forward)
            selected_shape = shapes[shape_type]
            
            noise_start, generated_shape, _ = generate_shape(forward, denoiser, selected_shape)
            
            # Visualization
            if show_comparison:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                axes[0].scatter(noise_start[:, 0], noise_start[:, 1], s=20, alpha=0.7, c='gray')
                axes[0].set_title('Starting Noise', fontsize=14)
                axes[0].set_xlim(-2, 2)
                axes[0].set_ylim(-2, 2)
                axes[0].grid(True, alpha=0.3)
                
                axes[1].scatter(generated_shape[:, 0], generated_shape[:, 1], s=20, alpha=0.7, c='green')
                axes[1].set_title('Generated Shape', fontsize=14)
                axes[1].set_xlim(-1.5, 1.5)
                axes[1].set_ylim(-1.5, 1.5)
                axes[1].grid(True, alpha=0.3)
                
                axes[2].scatter(selected_shape[:, 0], selected_shape[:, 1], s=20, alpha=0.7, c='red')
                axes[2].set_title('Original Shape', fontsize=14)
                axes[2].set_xlim(-1, 1)
                axes[2].set_ylim(-1, 1)
                axes[2].grid(True, alpha=0.3)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                
                axes[0].scatter(noise_start[:, 0], noise_start[:, 1], s=20, alpha=0.7, c='gray')
                axes[0].set_title('Starting Noise', fontsize=14)
                axes[0].set_xlim(-2, 2)
                axes[0].set_ylim(-2, 2)
                axes[0].grid(True, alpha=0.3)
                
                axes[1].scatter(generated_shape[:, 0], generated_shape[:, 1], s=20, alpha=0.7, c='green')
                axes[1].set_title('Generated Shape', fontsize=14)
                axes[1].set_xlim(-1.5, 1.5)
                axes[1].set_ylim(-1.5, 1.5)
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.info(f"📊 Generated {shape_type} using {num_steps} diffusion steps")

# Footer with explanation
st.markdown("---")
st.markdown("""
### 📚 How It Works

**Forward Process (Adding Noise)**
- Gradually adds Gaussian noise to the original shape over multiple steps
- Formula: `x_t = √(α̅_t) * x_0 + √(1-α̅_t) * ε`

**Reverse Process (Removing Noise)**  
- Starts with pure noise and removes it step-by-step
- In real models, a neural network (U-Net) learns to predict the noise

**Key Insight**: Real diffusion models like Stable Diffusion and DALL-E use this same principle but with:
- Deep neural networks for denoising
- Thousands of steps
- High-dimensional image data
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Increase steps for smoother transitions
- Try different shapes to see how noise affects them
- The denoiser here is simplified - real models are much better!
""")