import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_biggan_image(class_id, truncation=0.4):
    """
    Generate image using BigGAN with correct signature handling
    """
    # Load the model
    module = hub.load('https://tfhub.dev/deepmind/biggan-deep-256/1')
    
    # Create inputs with proper naming
    tf.random.set_seed(42)
    z = tf.random.normal([1, 128])  # Noise vector
    y = tf.one_hot([class_id], 1000)  # Class label
    truncation = tf.constant(truncation, dtype=tf.float32)
    
    # Get the model signature
    generator = module.signatures['default']
    
    # Generate with named parameters
    outputs = generator(z=z, y=y, truncation=truncation)
    
    # Process the output image
    image = outputs['default']
    image = tf.clip_by_value(image, -1, 1)
    image = (image + 1) / 2.0
    
    return image.numpy()

def display_image(image_array, title="Generated Image"):
    """Display the generated image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array[0])
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Vehicle class IDs
CLASS_IDS = {
    'sports_car': 817,
    'bus': 779,
    'taxi': 468,
    'pickup_truck': 717,
    'fire_engine': 555,
    'ambulance': 407,
    'limousine': 654,
    'minivan': 656
}

print("Generating vehicle images using BigGAN...")

try:
    # Generate sports car
    print(f"Generating sports car (Class {CLASS_IDS['sports_car']})...")
    sports_car = generate_biggan_image(CLASS_IDS['sports_car'], truncation=0.4)
    display_image(sports_car, "Sports Car")
    
    # Generate bus
    print(f"Generating bus (Class {CLASS_IDS['bus']})...")
    bus = generate_biggan_image(CLASS_IDS['bus'], truncation=0.4)
    display_image(bus, "Bus")
    
    # Generate taxi
    print(f"Generating taxi (Class {CLASS_IDS['taxi']})...")
    taxi = generate_biggan_image(CLASS_IDS['taxi'], truncation=0.4)
    display_image(taxi, "Taxi")
    
except Exception as e:
    print(f"Error: {e}")

print("Generation complete!")