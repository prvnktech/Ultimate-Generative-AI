# Chapter 06: Conditional GANs (cGANs)

This chapter introduces Conditional Generative Adversarial Networks (cGANs), which extend the basic GAN framework to enable controlled generation by conditioning on additional information such as class labels or other data.

## Overview

While standard GANs generate random samples, Conditional GANs allow you to specify what to generate. This chapter covers:

- **Conditional Generation** - Control the output by providing labels or conditions
- **Architecture Modifications** - Adding conditioning to Generator and Discriminator
- **Multi-Modal Generation** - Generate diverse outputs for the same condition
- **Practical Applications** - Text-to-image, class-conditional generation

## Features

- **Two Interactive Exercises**:
  - Exercise 6.1: Basic Conditional GAN implementation
  - Exercise 6.2: Advanced conditioning techniques
- **Streamlit Applications**: Interactive interfaces to generate specific types of images
- **Class-Conditional Generation**: Generate images of specific digits, objects, or categories
- **Label-Guided Synthesis**: Control generation with semantic labels

## Files

- `streamlit_chapter_06_exercise_6_1.py` - Basic cGAN implementation
- `streamlit_chapter_06_exercise_6_2.py` - Advanced cGAN techniques
- `Notebook/chapter_06_exercise_6_1.ipynb` - Detailed notebook for Exercise 6.1
- `Notebook/chapter_06_exercise_6_2.ipynb` - Detailed notebook for Exercise 6.2
- `gan_checkpoint.pth` - Pre-trained model checkpoint

## Installation

```bash
pip install torch torchvision streamlit matplotlib numpy pillow
```

## Usage

### Exercise 6.1: Basic Conditional GAN

```bash
streamlit run Chapter_06_Conditional_GANs/streamlit_chapter_06_exercise_6_1.py
```

Features:
- Generate images conditioned on class labels
- Train on MNIST with digit labels
- Select specific digits to generate
- Visualize conditioning effects

### Exercise 6.2: Advanced Conditional GAN

```bash
streamlit run Chapter_06_Conditional_GANs/streamlit_chapter_06_exercise_6_2.py
```

Features:
- Multi-modal conditioning
- Attribute-based generation
- Fine-grained control
- Quality improvements

### Jupyter Notebooks

```bash
jupyter notebook Chapter_06_Conditional_GANs/Notebook/
```

## Key Concepts

### How Conditioning Works

In a standard GAN:
- Generator: `G(z)` → image
- Discriminator: `D(x)` → probability

In a Conditional GAN:
- Generator: `G(z, y)` → image conditioned on label y
- Discriminator: `D(x, y)` → probability that x is real AND matches label y

### Conditional Generator Architecture

```
Input: Noise z + Condition y (concatenated or embedded)
↓
Embedding Layer (for categorical conditions)
↓
Concatenate noise and embedded condition
↓
Dense Layer + Reshape
↓
ConvTranspose2d + BatchNorm + ReLU
↓
ConvTranspose2d + BatchNorm + ReLU
↓
ConvTranspose2d + Tanh
↓
Output: Image matching the condition
```

### Conditional Discriminator Architecture

```
Input: Image x + Condition y
↓
Conv2d on image + Embed condition
↓
Concatenate feature maps with condition
↓
Conv2d + BatchNorm + LeakyReLU
↓
Conv2d + BatchNorm + LeakyReLU
↓
Flatten + Dense + Sigmoid
↓
Output: Real/Fake AND correct label probability
```

## What You'll Learn

1. **Conditioning Techniques**
   - Label embedding
   - Concatenation strategies
   - Conditional batch normalization
   - Projection discriminator

2. **Architecture Design**
   - Where to inject conditions
   - Embedding dimension choices
   - Feature fusion methods
   - Multi-scale conditioning

3. **Training Strategies**
   - Balanced conditioning
   - Label smoothing for conditional GANs
   - Handling class imbalance
   - Multi-task discriminator

4. **Applications**
   - Digit generation (MNIST)
   - Object generation (CIFAR-10)
   - Face generation with attributes
   - Text-to-image synthesis

## Types of Conditioning

### 1. Class Labels
Generate specific categories (e.g., "cat", "dog", "car")

### 2. Attributes
Control specific features (e.g., "smiling", "glasses", "age")

### 3. Text Descriptions
Generate images from text prompts

### 4. Spatial Conditions
Semantic segmentation maps, sketches, edges

### 5. Continuous Values
Control continuous attributes (e.g., pose angle, color intensity)

## Training Tips

### Effective Conditioning

- **Embedding Size**: Use 50-100 dimensions for class labels
- **Injection Points**: Add conditions at multiple layers
- **Normalization**: Normalize continuous conditions to [-1, 1]
- **Label Representation**: Use one-hot encoding for categorical data

### Avoiding Common Issues

| Issue | Solution |
|-------|----------|
| Generator ignores conditions | Increase conditioning strength, add auxiliary classifier |
| Mode collapse per class | Increase diversity loss, use minibatch discrimination |
| Poor condition-image alignment | Improve discriminator conditioning, add reconstruction loss |
| Imbalanced class generation | Use weighted sampling, class-aware training |

## Applications

### Image Synthesis
- **Digit Generation**: Generate specific numbers
- **Object Creation**: Create objects from categories
- **Face Synthesis**: Generate faces with desired attributes

### Image-to-Image Translation
- **Colorization**: Black & white → Color
- **Semantic Synthesis**: Segmentation map → Photo
- **Sketch-to-Photo**: Line drawing → Realistic image

### Data Augmentation
- Generate rare class samples
- Balance imbalanced datasets
- Create variations with controlled attributes

### Creative Applications
- Art generation by style
- Fashion design with specifications
- Architecture visualization

## Model Variants

### Auxiliary Classifier GAN (AC-GAN)
- Discriminator predicts both real/fake AND class label
- Encourages better class separation
- Improved sample quality

### Projection Discriminator
- Projects embedded condition into discriminator features
- More efficient conditioning
- Better performance on complex datasets

### StyleGAN with Conditioning
- Combines style-based generation with conditions
- Fine-grained control
- State-of-the-art quality

## Evaluation Metrics

- **Class Accuracy**: Can a classifier correctly identify generated image labels?
- **Conditional Inception Score**: IS computed per class
- **Conditional FID**: FID score for each condition separately
- **Label Consistency**: Alignment between condition and generated content

## Example Use Case: MNIST Digit Generation

```python
# Generate specific digit (e.g., digit 7)
condition = torch.tensor([7])  # Class label
noise = torch.randn(1, latent_dim)
generated_image = generator(noise, condition)
```

## Advanced Topics

- **Multi-Modal Conditioning**: Multiple types of conditions simultaneously
- **Hierarchical Conditioning**: Coarse-to-fine control
- **Disentangled Conditioning**: Independent control of different attributes
- **Zero-Shot Generation**: Generate unseen class combinations

## Next Steps

After mastering Conditional GANs:
- **Chapter 07**: CycleGANs - Unpaired image-to-image translation
- **Chapter 08**: StyleGANs - Advanced style-based generation
- **Pix2Pix**: Paired image-to-image translation with cGANs

## Resources

- [Conditional GAN (Original Paper)](https://arxiv.org/abs/1411.1784)
- [AC-GAN Paper](https://arxiv.org/abs/1610.09585)
- [Projection Discriminator](https://arxiv.org/abs/1802.05637)
- [cGAN Tutorial - TensorFlow](https://www.tensorflow.org/tutorials/generative/cgan)

---

**Start Generating!** Launch the Streamlit apps and create images on demand!
