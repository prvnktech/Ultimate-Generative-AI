#!/usr/bin/env python3
"""
chapter_01_biggan_car.py
Generate images of vehicles (sports car, taxi, bus ...) using a pretrained BigGAN from TensorFlow Hub.

Example:
    python Chapter_01_Introduction/scripts/chapter_01_biggan_car.py \
        --class_name sports_car --output outputs/sports_car.png --seed 42
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

# -------------------------
# Class name -> ImageNet class index (examples from repo docs)
CLASS_MAP = {
    "sports_car": 817,   # sports car (per your repo docs)
    "bus": 779,
    "taxi": 468,
    # add more mappings if you want
}

# Default TF Hub model handle (change if you have a different/preferred BigGAN)
DEFAULT_MODEL_HANDLE = "https://tfhub.dev/deepmind/biggan-deep-256/1"

# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BigGAN vehicle image generator (Chapter 1 example)")
    p.add_argument("--class_name", type=str, default="sports_car",
                   help="Vehicle class to generate: sports_car, bus, taxi, ...")
    p.add_argument("--output", type=str, default="outputs/sports_car.png",
                   help="Output image path")
    p.add_argument("--model_handle", type=str, default=DEFAULT_MODEL_HANDLE,
                   help="TensorFlow Hub model handle (override if needed)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible outputs")
    p.add_argument("--truncation", type=float, default=0.4,
                   help="Truncation value (typical range 0.4-1.0). Lower = cleaner but less variety.")
    return p.parse_args()


# -------------------------
def make_one_hot(class_idx, num_classes=1000):
    y = np.zeros((1, num_classes), dtype=np.float32)
    y[0, class_idx] = 1.0
    return y


# -------------------------
def sample_z(batch_size=1, dim=128, truncation=1.0, seed=None):
    """Sample latent vector z using truncated normal distribution."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    # truncated normal: sample from normal then clip to +/- truncation*stddev
    z = rng.normal(size=(batch_size, dim)).astype(np.float32)
    if truncation < 1.0:
        z = np.clip(z, -truncation, truncation)
    return z


# -------------------------
def postprocess_and_save(image_tensor, output_path):
    """
    image_tensor: numpy array with shape [1, H, W, 3], values in [-1, 1] (common for BigGAN)
    """
    img = image_tensor[0]
    # Convert from [-1,1] to [0,255]
    img = (img + 1.0) / 2.0
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pil.save(output_path)
    print(f"Saved image to: {output_path}")


# -------------------------
def main():
    args = parse_args()

    class_name = args.class_name.lower()
    if class_name not in CLASS_MAP:
        print("Unknown --class_name. Available options:", ", ".join(CLASS_MAP.keys()))
        return

    class_idx = CLASS_MAP[class_name]
    print(f"Generating class '{class_name}' (ImageNet idx {class_idx})")

    # Load model
    print("Loading BigGAN model from TF Hub:", args.model_handle)
    model = hub.load(args.model_handle)  # returns a callable

    # Prepare inputs
    z_dim = 128  # BigGAN latent dim (standard)
    z = sample_z(batch_size=1, dim=z_dim, truncation=args.truncation, seed=args.seed)
    y = make_one_hot(class_idx, num_classes=1000)

    # Convert to tensors
    z_tf = tf.convert_to_tensor(z, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    truncation_tf = tf.convert_to_tensor(args.truncation, dtype=tf.float32)

    # Run the model
    # API: many TF-Hub BigGAN modules accept (z, y, truncation) -> images
    # The returned images are typically floats in [-1, 1].
    print("Running model... (this may take a while on CPU)")
    images = model(z_tf, y_tf, truncation_tf)
    images_np = images.numpy()

    # Postprocess and save
    postprocess_and_save(images_np, args.output)


if __name__ == "__main__":
    main()
