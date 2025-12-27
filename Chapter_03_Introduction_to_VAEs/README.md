# Requirement 3.2 – Variational Autoencoder (VAE) using PyTorch

## Overview
This project demonstrates a **simple Variational Autoencoder (VAE)** implemented using **PyTorch**.  
The VAE is trained on the **MNIST dataset** and visualizes:

- Original input image
- Reconstructed image from the encoder–decoder
- Newly generated image sampled from the latent space

The implementation is intentionally kept **simple and educational**, suitable for:
- Jupyter Notebook
- VS Code / Local Python execution

---

## Features
- PyTorch-based VAE implementation
- Fully connected (Dense) encoder and decoder
- Reparameterization trick (μ, log σ²)
- Custom VAE loss (Reconstruction + KL Divergence)
- Visualization using Matplotlib

---

## Project Structure
```text
vae_pytorch.py
README.md
data/
