# Chapter 10: Diffusion Models

This chapter introduces Diffusion Models, one of the most powerful and recent approaches to generative modeling that has achieved state-of-the-art results in image generation, surpassing GANs in many benchmarks.

## Overview

Diffusion Models generate data by gradually denoising random noise through a learned reverse diffusion process. This chapter covers:

- **Forward Diffusion Process** - Gradually adding noise to data
- **Reverse Diffusion Process** - Learning to remove noise step-by-step
- **Denoising Diffusion Probabilistic Models (DDPM)** - Core architecture
- **Score-Based Models** - Alternative perspective on diffusion
- **Sampling Techniques** - Efficient generation methods

## Features

- **Interactive Exercise**: Comprehensive diffusion model implementation
- **Streamlit Application**: Visualize the diffusion and denoising process
- **Step-by-Step Generation**: Watch images emerge from noise
- **Noise Schedule Exploration**: Experiment with different noise schedules
- **Quality Comparison**: Compare with GANs and VAEs

## Files

- `streamlit_chapter_10_exercise_10.1.py` - Interactive diffusion model demo
- `Notebook/chapter_10.ipynb` - Detailed theoretical and practical notebook

## Installation

```bash
pip install torch torchvision streamlit matplotlib numpy pillow tqdm
```

Optional (for advanced features):
```bash
pip install diffusers transformers accelerate
```

## Usage

### Streamlit Application

```bash
streamlit run Chapter_10_Diffusion_Models/streamlit_chapter_10_exercise_10.1.py
```

Features:
- Visualize forward diffusion (noise addition)
- Train denoising network
- Generate samples from noise
- Adjust noise schedules (linear, cosine)
- Compare different timesteps

### Jupyter Notebook

```bash
jupyter notebook Chapter_10_Diffusion_Models/Notebook/chapter_10.ipynb
```

## Key Concepts

### Forward Diffusion Process

Gradually add Gaussian noise over T timesteps:

```
x₀ (original image) → x₁ → x₂ → ... → xₜ (noisy) → ... → xₜ (pure noise)
```

**Formula:**
```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)·xₜ₋₁, βₜI)
```

Where:
- βₜ: Variance schedule (controls noise amount)
- xₜ: Image at timestep t
- T: Total timesteps (typically 1000)

### Reverse Diffusion Process

Learn to remove noise step by step:

```
xₜ (noise) → xₜ₋₁ → ... → x₁ → x₀ (generated image)
```

**Neural Network learns:**
```
pθ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))
```

### Denoising Network Architecture

```
Input: Noisy image xₜ + Timestep t
↓
Time Embedding (sinusoidal position encoding)
↓
U-Net Architecture:
  - Encoder: Downsample with ResNet blocks
  - Bottleneck: Self-attention layers
  - Decoder: Upsample with skip connections
↓
Output: Predicted noise εθ(xₜ, t)
```

### Training Objective

**Loss Function (Simplified):**
```
L = 𝔼ₜ,x₀,ε [||ε - εθ(xₜ, t)||²]
```

Where:
- ε: True noise added to image
- εθ(xₜ, t): Predicted noise by network
- Goal: Predict the noise that was added

## What You'll Learn

1. **Diffusion Theory**
   - Markov chain formulation
   - Forward and reverse processes
   - Variational lower bound
   - Connection to score matching

2. **Implementation Details**
   - U-Net architecture for denoising
   - Time step embeddings
   - Noise scheduling strategies
   - Training procedures

3. **Sampling Methods**
   - DDPM sampling (slow but high quality)
   - DDIM sampling (faster, deterministic)
   - Accelerated sampling techniques
   - Conditional generation

4. **Advanced Techniques**
   - Classifier guidance
   - Classifier-free guidance
   - Latent diffusion models
   - Text-to-image diffusion

## Noise Schedules

### Linear Schedule
```python
βₜ = β_min + (β_max - β_min) * t/T
```
- Simple and commonly used
- β_min = 0.0001, β_max = 0.02

### Cosine Schedule
```python
αₜ = cos((t/T + s)/(1 + s) * π/2)²
```
- Better for high resolution images
- More stable training

### Learned Schedule
- Network learns optimal noise schedule
- Dataset-dependent optimization

## Advantages of Diffusion Models

| Aspect | Diffusion Models | GANs | VAEs |
|--------|------------------|------|------|
| **Training Stability** | Very Stable | Can be unstable | Stable |
| **Sample Quality** | Excellent | Excellent | Good |
| **Sample Diversity** | High | Can have mode collapse | High |
| **Likelihood** | Can compute | Cannot compute | Can compute |
| **Speed** | Slow (many steps) | Fast (one pass) | Fast (one pass) |
| **Architecture** | U-Net + attention | Two networks | Encoder-decoder |

## Training Tips

### Hyperparameters

- **Timesteps (T)**: 1000 is standard
- **Learning Rate**: 1e-4 to 2e-4
- **Batch Size**: As large as GPU allows (64-128)
- **Model Size**: 50-100M parameters for good quality
- **Training Steps**: 500K-1M for convergence

### Best Practices

1. **Use EMA (Exponential Moving Average)** of model weights
2. **Normalize images** to [-1, 1] range
3. **Use self-attention** at lower resolutions (16x16, 8x8)
4. **Gradient clipping** to stabilize training
5. **Mixed precision training** for efficiency

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Slow convergence | High loss after many steps | Increase model capacity, check learning rate |
| Poor sample quality | Blurry or artifacts | Train longer, use better noise schedule |
| Mode collapse | Limited diversity | Rare with diffusion, check data pipeline |
| Memory issues | OOM errors | Reduce batch size, use gradient checkpointing |

## Sampling Strategies

### DDPM (Original)
- 1000 steps for best quality
- Stochastic sampling
- ~20 seconds per image on GPU

### DDIM (Faster)
- 50-100 steps for similar quality
- Deterministic sampling
- ~2 seconds per image on GPU

### Progressive Distillation
- Train student model to skip steps
- 4-8 steps for generation
- Minimal quality loss

## Applications

### Image Generation
- **High-Quality Synthesis**: State-of-the-art results
- **Super-Resolution**: Enhance low-res images
- **Inpainting**: Fill missing regions
- **Unconditional Generation**: Generate diverse samples

### Conditional Generation
- **Text-to-Image**: DALL-E 2, Imagen, Stable Diffusion
- **Class-Conditional**: Generate specific categories
- **Image-to-Image**: Guided editing and translation

### Other Domains
- **Audio Synthesis**: Music and speech generation
- **Video Generation**: Temporal consistency
- **3D Generation**: Shapes and textures
- **Molecule Design**: Drug discovery

## Notable Models

### DALL-E 2 (OpenAI)
- Text-to-image using CLIP guidance
- High quality and editability

### Imagen (Google)
- Large language model conditioning
- Photorealistic generation

### Stable Diffusion (Stability AI)
- Latent diffusion in compressed space
- Open-source and efficient
- Powers many applications

### Midjourney
- Artistic image generation
- Discord-based interface

## Latent Diffusion Models

Run diffusion in **latent space** instead of pixel space:

```
Image → VAE Encoder → Latent → Diffusion → VAE Decoder → Image
```

**Advantages:**
- Much faster (smaller dimensions)
- Lower memory requirements
- Similar quality to pixel-space diffusion

**Example: Stable Diffusion**
- Operates on 64x64 latent space
- Generates 512x512 images
- ~10x faster than pixel diffusion

## Guidance Techniques

### Classifier Guidance
```
Score = Score_unconditional + w * ∇log p(y|x)
```
- Use classifier gradients to guide generation
- Requires training classifier

### Classifier-Free Guidance
```
Score = Score_unconditional + w * (Score_conditional - Score_unconditional)
```
- No separate classifier needed
- Train model with and without conditioning
- More commonly used

## Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Measures realism and diversity
- **IS (Inception Score)**: Measures quality and diversity
- **Precision/Recall**: Quality vs diversity trade-off
- **CLIP Score**: Text-image alignment (for conditional models)

## Code Example: Simple Diffusion

```python
# Forward diffusion (add noise)
noisy_image = sqrt(alpha_t) * image + sqrt(1 - alpha_t) * noise

# Train denoising network
predicted_noise = model(noisy_image, timestep)
loss = MSE(predicted_noise, noise)

# Sampling (reverse diffusion)
x = random_noise
for t in reversed(range(T)):
    x = denoise_step(x, t)
generated_image = x
```

## Advanced Topics

- **Score-Based Generative Models**: SDE formulation
- **Continuous-Time Diffusion**: ODE solvers
- **Improved DDPM**: Learned variances, hybrid objectives
- **Cascaded Diffusion**: Multi-resolution generation
- **Diffusion for Audio/Video**: Temporal modeling

## Next Steps

After mastering Diffusion Models:
- **Chapter 11**: Data Augmentation with generative models
- **Chapter 12**: Generative Models in NLP
- **Explore**: Stable Diffusion, DALL-E alternatives

## Resources

- [DDPM Paper (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239)
- [DDIM Paper (Denoising Diffusion Implicit Models)](https://arxiv.org/abs/2010.02502)
- [Stable Diffusion](https://stability.ai/stable-diffusion)
- [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers)
- [Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

**Start Diffusing!** Launch the Streamlit app and watch images emerge from noise!
