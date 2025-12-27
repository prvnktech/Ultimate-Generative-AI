# Chapter 11: Data Augmentation with Generative Models

This chapter explores how generative models can be used for data augmentation, helping to improve machine learning model performance by creating diverse synthetic training data.

## Overview

Data augmentation is crucial for training robust machine learning models, especially when labeled data is scarce. This chapter demonstrates:

- **GAN-Based Augmentation** - Generate synthetic training samples
- **VAE-Based Augmentation** - Create variations of existing data
- **Augmentation Strategies** - When and how to use synthetic data
- **Quality vs Quantity** - Balancing synthetic and real data
- **Domain-Specific Techniques** - Image, text, and tabular data augmentation

## Features

- **Practical Implementations**: Ready-to-use augmentation pipelines
- **Jupyter Notebooks**: Detailed explanations with examples
- **Multiple Modalities**: Image and data augmentation techniques
- **Performance Analysis**: Compare models with and without augmentation
- **Best Practices**: Guidelines for effective augmentation

## Files

- `11.1.py` - GAN-based data augmentation implementation
- `11.2.py` - VAE-based data augmentation implementation
- `Notebook/11.1.ipynb` - GAN augmentation detailed notebook
- `Notebook/11.2.ipynb` - VAE augmentation detailed notebook

## Installation

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

For advanced features:
```bash
pip install albumentations imgaug
```

## Usage

### GAN-Based Augmentation

```bash
python Chapter_11_Data_Augmentation/11.1.py
```

Features:
- Train GAN on limited dataset
- Generate synthetic samples
- Augment training set
- Evaluate model performance

### VAE-Based Augmentation

```bash
python Chapter_11_Data_Augmentation/11.2.py
```

Features:
- Learn latent representations
- Sample variations from latent space
- Create diverse augmented dataset
- Compare reconstruction quality

### Jupyter Notebooks

```bash
jupyter notebook Chapter_11_Data_Augmentation/Notebook/
```

## Key Concepts

### Why Data Augmentation?

**Problems with Limited Data:**
- Overfitting to training set
- Poor generalization
- Bias toward common patterns
- Insufficient coverage of variations

**Solutions with Generative Models:**
- Create unlimited synthetic samples
- Generate rare/edge cases
- Balance imbalanced datasets
- Explore data manifold

### Traditional vs Generative Augmentation

| Technique | Traditional | Generative |
|-----------|-------------|------------|
| **Methods** | Rotation, flip, crop, color jitter | GAN/VAE generation |
| **Diversity** | Limited transformations | High diversity |
| **Realism** | Always realistic (real image based) | Can be unrealistic if poorly trained |
| **Control** | Easy to control | Harder to control specific augmentations |
| **Cost** | Fast and cheap | Requires training generative model |
| **Use Case** | Always useful | Best when data is very limited |

## Augmentation Strategies

### 1. GAN-Based Augmentation

**Process:**
```
1. Train GAN on original dataset
2. Generate synthetic samples
3. Mix synthetic + real data
4. Train target model on augmented dataset
```

**Advantages:**
- High-quality realistic samples
- Can generate entirely new examples
- Good for complex distributions

**Considerations:**
- Requires sufficient data to train GAN
- Generated samples must be diverse
- Avoid mode collapse

### 2. VAE-Based Augmentation

**Process:**
```
1. Train VAE on original dataset
2. Encode real samples to latent space
3. Sample nearby points in latent space
4. Decode to create variations
5. Use augmented dataset for training
```

**Advantages:**
- Stable training
- Controllable variations
- Smooth interpolations

**Considerations:**
- Samples may be blurrier than GANs
- Limited to variations of existing data
- Requires good latent space coverage

### 3. Hybrid Approaches

Combine multiple techniques:
- Traditional augmentation + GAN generation
- VAE interpolation + traditional transforms
- Multi-model ensemble augmentation

## What You'll Learn

1. **Augmentation Techniques**
   - When to use generative augmentation
   - Choosing GAN vs VAE for augmentation
   - Mixing ratios (synthetic/real data)
   - Quality control for generated data

2. **Implementation Strategies**
   - Training generative models for augmentation
   - Sampling strategies
   - Data balancing techniques
   - Pipeline integration

3. **Evaluation Methods**
   - Measuring augmentation effectiveness
   - Diversity metrics
   - Downstream task performance
   - Overfitting detection

4. **Domain-Specific Applications**
   - Image augmentation
   - Tabular data augmentation
   - Time series augmentation
   - Text augmentation

## Data Augmentation Pipeline

```python
# Step 1: Train generative model
gan = train_gan(limited_dataset)

# Step 2: Generate synthetic data
synthetic_data = gan.generate(num_samples=10000)

# Step 3: Filter quality
high_quality = filter_samples(synthetic_data, threshold=0.8)

# Step 4: Mix with real data
augmented_dataset = combine(real_data, high_quality, ratio=0.5)

# Step 5: Train target model
classifier = train_model(augmented_dataset)
```

## Best Practices

### Training Generative Models

1. **Use enough real data**: At least 1000-5000 samples per class
2. **Validate generated quality**: Visual inspection and metrics
3. **Monitor diversity**: Ensure mode coverage
4. **Use pretrained models**: Transfer learning when possible

### Using Generated Data

1. **Start with small mixing ratios**: 10-30% synthetic data
2. **Gradually increase**: Monitor validation performance
3. **Filter low-quality samples**: Use discriminator scores
4. **Balance classes**: Equal synthetic samples per class

### Quality Control

```python
# Discriminator-based filtering
def filter_quality(generated_samples, discriminator, threshold=0.7):
    scores = discriminator(generated_samples)
    return generated_samples[scores > threshold]

# Diversity checking
def check_diversity(samples, min_distance=0.1):
    # Ensure samples are sufficiently different
    pairwise_distances = compute_distances(samples)
    return pairwise_distances.mean() > min_distance
```

## Applications

### Image Classification
- **Medical Imaging**: Generate rare disease cases
- **Autonomous Driving**: Create diverse traffic scenarios
- **Quality Inspection**: Synthesize defect examples

### Object Detection
- Generate objects in different poses
- Create varied lighting conditions
- Synthesize occluded objects

### Imbalanced Datasets
- Oversample minority classes
- Balance class distribution
- Prevent bias

### Few-Shot Learning
- Generate examples from few samples
- Meta-learning augmentation
- Domain adaptation

### Privacy-Preserving ML
- Generate synthetic data instead of using real data
- Maintain statistical properties
- Enable data sharing

## Metrics for Evaluation

### Generative Model Quality
- **FID (Fréchet Inception Distance)**: Realism measure
- **IS (Inception Score)**: Quality and diversity
- **Precision/Recall**: Coverage vs quality

### Augmentation Effectiveness
- **Downstream Accuracy**: Target task performance
- **Generalization Gap**: Train-test difference
- **Robustness**: Performance on perturbed data
- **Calibration**: Confidence alignment

## Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Low-quality generations | Noise in training | Use discriminator filtering |
| Mode collapse | Low diversity | Use better GAN architecture |
| Distribution shift | Poor generalization | Validate on real test set |
| Over-reliance on synthetic | Model learns artifacts | Limit synthetic data ratio |
| Computational cost | Slow training | Use pretrained models |

## Advanced Techniques

### Conditional Augmentation
Generate specific variations:
```python
# Generate more samples of class X
synthetic_class_x = cgan.generate(label=X, num_samples=1000)
```

### Progressive Augmentation
Start with traditional, add generative:
```python
# Traditional first
augmented = traditional_augment(data)
# Add generative if still insufficient
if len(augmented) < threshold:
    augmented += gan.generate(shortage)
```

### Meta-Learning for Augmentation
Learn which augmentations work best:
```python
# Optimize augmentation strategy
best_strategy = meta_learner.optimize(augmentation_policies)
```

### Adversarial Augmentation
Generate hard examples:
```python
# Create challenging samples
hard_samples = gan.generate_adversarial(classifier)
```

## Domain-Specific Guidelines

### Medical Imaging
- Careful validation by experts
- Focus on rare pathologies
- Maintain anatomical correctness
- Consider regulatory requirements

### Natural Images
- Preserve semantic meaning
- Maintain realistic textures
- Avoid unrealistic combinations
- Check for memorization

### Tabular Data
- Preserve correlations between features
- Maintain constraint satisfaction
- Handle categorical and continuous features
- Validate statistical properties

## Code Examples

### Example 1: Image Augmentation with GAN

```python
from torchvision import transforms

# Train GAN
gan = DCGAN(latent_dim=100)
gan.train(real_images, epochs=100)

# Generate synthetic images
z = torch.randn(1000, 100)
synthetic = gan.generate(z)

# Combine with real data
augmented_dataset = torch.cat([real_images, synthetic], dim=0)
```

### Example 2: VAE Interpolation

```python
# Encode real samples
z1 = vae.encode(image1)
z2 = vae.encode(image2)

# Interpolate in latent space
alphas = np.linspace(0, 1, 10)
interpolated = [vae.decode((1-α)*z1 + α*z2) for α in alphas]
```

## Next Steps

After mastering Data Augmentation:
- **Chapter 12**: Generative Models in NLP
- **Chapter 16**: Introduction to Large Language Models
- **Explore**: AutoML for augmentation strategies

## Resources

- [A Survey on Data Augmentation](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
- [GAN Augmentation Paper](https://arxiv.org/abs/1711.04340)
- [Albumentations Library](https://albumentations.ai/)
- [imgaug Documentation](https://imgaug.readthedocs.io/)
- [Data Augmentation Review](https://arxiv.org/abs/1904.12848)

---

**Augment Your Data!** Use generative models to boost your ML performance!
