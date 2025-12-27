# Chapter 09: Advanced VAEs

This chapter explores advanced Variational Autoencoder (VAE) architectures and techniques that improve generation quality, disentanglement, and controllability.

## Overview

Building on the basic VAE concepts from Chapter 3, this chapter introduces state-of-the-art VAE variants:

- **β-VAE** - Disentangled representation learning
- **VQ-VAE** - Vector Quantized VAE for discrete latent spaces
- **Hierarchical VAEs** - Multi-scale latent representations
- **Conditional VAEs** - Controlled generation with labels

## Features

- **Two Interactive Exercises**:
  - Exercise 9.1: β-VAE and disentanglement
  - Exercise 9.2: VQ-VAE and discrete representations
- **Streamlit Applications**: Interactive exploration of advanced VAE concepts
- **Disentanglement Visualization**: See how different latent dimensions control specific features
- **Latent Space Traversal**: Navigate and manipulate learned representations

## Files

- `streamlit_chapter_09_exercise_9_1.py` - β-VAE implementation
- `streamlit_chapter_09_exercise_9_2.py` - VQ-VAE implementation
- `Notebook/chapter_09_exercise_9_1.ipynb` - Detailed notebook for Exercise 9.1
- `Notebook/chapter_09_exercise_9_2.ipynb` - Detailed notebook for Exercise 9.2
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r Chapter_09_Advanced_VAEs/requirements.txt
```

Or install manually:
```bash
pip install torch torchvision streamlit matplotlib numpy pillow scikit-learn
```

## Usage

### Exercise 9.1: β-VAE

```bash
streamlit run Chapter_09_Advanced_VAEs/streamlit_chapter_09_exercise_9_1.py
```

Features:
- Train β-VAE with adjustable β parameter
- Visualize disentangled factors
- Traverse individual latent dimensions
- Compare with standard VAE

### Exercise 9.2: VQ-VAE

```bash
streamlit run Chapter_09_Advanced_VAEs/streamlit_chapter_09_exercise_9_2.py
```

Features:
- Vector quantization mechanism
- Codebook learning
- Discrete latent representations
- Reconstruction quality analysis

### Jupyter Notebooks

```bash
jupyter notebook Chapter_09_Advanced_VAEs/Notebook/
```

## Key Concepts

### 1. β-VAE (Beta-VAE)

Introduces a hyperparameter β to balance reconstruction and disentanglement:

**Loss Function:**
```
L = Reconstruction_Loss + β * KL_Divergence
```

**Properties:**
- β = 1: Standard VAE
- β > 1: Encourages disentanglement (at cost of reconstruction)
- β < 1: Better reconstruction (less disentanglement)

**Applications:**
- Interpretable representations
- Feature manipulation
- Fair representation learning
- Controllable generation

### 2. VQ-VAE (Vector Quantized VAE)

Uses discrete latent representations instead of continuous:

**Architecture:**
```
Encoder → Continuous latent → Vector Quantization → Discrete latent
Discrete latent → Decoder → Reconstruction
```

**Key Components:**
- **Codebook**: Learned discrete vectors
- **Vector Quantization**: Map encoder outputs to nearest codebook entry
- **Commitment Loss**: Encourage encoder to commit to codebook entries

**Advantages:**
- No posterior collapse
- Better long-range dependencies
- Compatible with autoregressive priors
- Useful for downstream tasks

### 3. Hierarchical VAEs

Multi-level latent representations for complex data:

```
Image → Encoder
  ↓
High-level latent (semantic features)
  ↓
Mid-level latent (object parts)
  ↓
Low-level latent (textures, edges)
  ↓
Decoder → Reconstruction
```

**Benefits:**
- Capture features at multiple scales
- Better model complex distributions
- Improved sample quality

### 4. Conditional VAEs (CVAEs)

Condition generation on labels or attributes:

```
Encoder: (x, y) → z
Decoder: (z, y) → x̂
```

**Use Cases:**
- Class-conditional generation
- Attribute manipulation
- Semi-supervised learning
- Multi-modal generation

## What You'll Learn

1. **Disentanglement Theory**
   - What is disentanglement?
   - Measuring disentanglement
   - β-VAE objective and trade-offs
   - Interpretable latent factors

2. **Discrete Representations**
   - Vector quantization mechanism
   - Codebook learning strategies
   - Straight-through estimator
   - VQ-VAE applications

3. **Advanced Training Techniques**
   - KL annealing
   - Warm-up strategies
   - Balancing reconstruction and regularization
   - Avoiding posterior collapse

4. **Evaluation Metrics**
   - Reconstruction quality (MSE, SSIM)
   - Disentanglement metrics (MIG, SAP, DCI)
   - Sample quality (IS, FID)
   - Latent traversal analysis

## Comparison: VAE Variants

| Model | Latent Space | Disentanglement | Quality | Complexity |
|-------|--------------|-----------------|---------|------------|
| VAE | Continuous | Low | Good | Low |
| β-VAE | Continuous | High | Medium | Low |
| VQ-VAE | Discrete | Medium | High | Medium |
| Hierarchical VAE | Multi-level | Medium | High | High |
| CVAE | Conditional | Medium | Good | Medium |

## Training Tips

### β-VAE Training

- **Start with β = 1**, gradually increase
- Monitor reconstruction and KL divergence separately
- Optimal β depends on dataset and disentanglement goals
- Use β = 4-10 for good disentanglement

### VQ-VAE Training

- **Codebook size**: 64-512 entries typical
- **Commitment cost**: λ = 0.25 is standard
- Use exponential moving average (EMA) for codebook updates
- Initialize codebook with k-means clustering

### Avoiding Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Posterior Collapse | KL → 0, poor samples | Use KL annealing, reduce β initially |
| Poor Disentanglement | Entangled factors | Increase β, use better architecture |
| Codebook Collapse | Few codebook entries used | Increase commitment cost, restart unused codes |
| Blurry Reconstructions | High β, poor quality | Reduce β, improve decoder capacity |

## Applications

### Image Generation
- High-quality synthetic images
- Controllable generation
- Style transfer

### Representation Learning
- Feature extraction
- Dimensionality reduction
- Unsupervised learning

### Data Compression
- Lossy compression with VQ-VAE
- Efficient encoding

### Semi-Supervised Learning
- Leverage unlabeled data
- Conditional generation for data augmentation

### Drug Discovery
- Molecular generation
- Property optimization

## Disentanglement Metrics

### 1. Mutual Information Gap (MIG)
Measures how much a single latent variable captures a ground truth factor

### 2. Separated Attribute Predictability (SAP)
Evaluates how well individual latents predict specific factors

### 3. Disentanglement-Completeness-Informativeness (DCI)
Three-way metric for comprehensive evaluation

### 4. Factor-VAE Metric
Based on variance of latent distributions

## Example: Disentangled Face Generation

```python
# Generate faces with controlled attributes
z = sample_latent()  # Sample base latent code

# Modify specific dimensions
z[3] = -2.0  # Change smile
z[7] = 1.5   # Change age
z[12] = 0.8  # Change gender

generated_face = decoder(z)
```

## Advanced Topics

- **InfoVAE**: Maximize mutual information
- **FactorVAE**: Explicit disentanglement objective
- **TC-VAE**: Total correlation minimization
- **Wasserstein VAE**: Alternative divergence measure
- **Adversarial VAEs**: Combine VAE with adversarial training

## Datasets for Experiments

- **dSprites**: Simple shapes with controllable factors
- **3D Shapes**: 3D rendered objects
- **CelebA**: Face images with attributes
- **MNIST**: Handwritten digits
- **CIFAR-10**: Natural images

## Next Steps

After mastering Advanced VAEs:
- **Chapter 10**: Diffusion Models - Latest generation technique
- **Chapter 11**: Data Augmentation with generative models
- **Compare**: VAEs vs GANs vs Diffusion Models

## Resources

- [β-VAE Paper](https://openreview.net/forum?id=Sy2fzU9gl)
- [VQ-VAE Paper](https://arxiv.org/abs/1711.00937)
- [Disentanglement Review Paper](https://arxiv.org/abs/1812.02230)
- [VAE Tutorial](https://arxiv.org/abs/1606.05908)
- [Disentanglement Metrics](https://github.com/google-research/disentanglement_lib)

---

**Explore Advanced VAEs!** Launch the Streamlit apps and discover disentangled representations!
