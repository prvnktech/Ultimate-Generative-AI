# Chapter 01: Introduction to Generative Models

This chapter introduces the fundamental concepts of generative models and demonstrates how to build a simple random car image generator.

## Overview

Generative models are machine learning models that learn to generate new data samples similar to the training data. This chapter covers:

- **What are Generative Models?** - Understanding the core concepts
- **Types of Generative Models** - Overview of different approaches
- **Simple Generator Implementation** - Building a basic random generator
- **Image Generation** - Creating synthetic car images

## Features

- **Interactive Notebook**: Explore concepts with executable code
- **Streamlit Application**: Interactive web interface for generating car images
- **Visual Demonstrations**: Generate and visualize random car images
- **Hands-on Learning**: Understand the basics before diving into complex models

## Files

- `chapter_01_generator.py` - Python script for generating car images
- `streamlit_chapter_01_exercise_1.py` - Interactive Streamlit application
- `Notebook/chapter_01_generator.ipynb` - Jupyter notebook with detailed explanations
- `generated_car.png` - Example output from the generator

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Streamlit App

```bash
streamlit run Chapter_01_Introduction_to_Generative_Models/streamlit_chapter_01_exercise_1.py
```

The web interface will allow you to:
- Generate random car images
- Experiment with different parameters
- Visualize the generation process

### Option 2: Run Jupyter Notebook

```bash
jupyter notebook Chapter_01_Introduction_to_Generative_Models/Notebook/chapter_01_generator.ipynb
```

### Option 3: Run Python Script

```bash
python Chapter_01_Introduction_to_Generative_Models/chapter_01_generator.py
```

## Key Concepts

### Generative vs Discriminative Models

- **Discriminative Models**: Learn P(y|x) - predict labels from data
- **Generative Models**: Learn P(x|y) or P(x) - generate new data samples

### Simple Random Generator

This chapter implements a basic random generator that:
1. Uses random noise as input
2. Applies simple transformations
3. Generates car-like images

While simple, this introduces the core concept that will be expanded in later chapters with VAEs, GANs, and Diffusion Models.

## What You'll Learn

1. **Foundation Concepts**
   - Purpose of generative models
   - Difference from discriminative approaches
   - Real-world applications

2. **Practical Implementation**
   - Building a simple generator
   - Image manipulation techniques
   - Random sampling methods

3. **Visualization**
   - Displaying generated samples
   - Understanding output quality
   - Iterative improvements

## Next Steps

After completing this chapter, you'll be ready to explore:
- **Chapter 02**: Mathematical foundations (probability, entropy, KL divergence)
- **Chapter 03**: Variational Autoencoders (VAEs)
- **Chapter 04**: Generative Adversarial Networks (GANs)

## Applications

Generative models are used in:
- Image synthesis and editing
- Text generation
- Music composition
- Drug discovery
- Data augmentation
- Anomaly detection

## Resources

- [Generative Models - Stanford CS236](https://deepgenerativemodels.github.io/)
- [Introduction to Generative Models](https://developers.google.com/machine-learning/gan/generative)
- [Deep Generative Modeling - MIT](https://deepgenerativemodels.github.io/)

---

**Ready to start?** Launch the Streamlit app and generate your first images!
