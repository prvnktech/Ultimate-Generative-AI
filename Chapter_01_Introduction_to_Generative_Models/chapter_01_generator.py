import os
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

# 1. Disable TensorFlow 2 behaviors for BigGAN compatibility
tf.disable_v2_behavior()

def run_biggan():
    # Model URL for 256x256 resolution
    module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'
    
    print(f"Loading BigGAN module from: {module_path}")
    # Load the module using legacy TF1 Hub API
    module = hub.Module(module_path)
    
    # Prepare inputs: z (noise), y (class), and truncation
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k) 
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)

    # ImageNet class 817 is "sports car"
    car_class = 817 
    truncation = 0.5
    
    # Create the class vector (one-hot encoding)
    y_data = np.zeros((1, 1000))
    y_data[0, car_class] = 1
    
    # Generate random noise vector
    z_data = np.random.normal(size=(1, 128))

    # Start a session and run the generator
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Generating image...")
        
        samples = sess.run(output, feed_dict={
            inputs['z']: z_data,
            inputs['y']: y_data,
            inputs['truncation']: truncation
        })

        # Post-process: Convert [-1, 1] range to [0, 255] uint8
        samples = (samples + 1.0) / 2.0
        img_array = (samples[0] * 255).astype(np.uint8)
        
        # Save and show the image
        img = PIL.Image.fromarray(img_array)
        img.save("generated_car.png")
        img.show()
        print("Success! Image saved as 'generated_car.png'")

if __name__ == "__main__":
    run_biggan()