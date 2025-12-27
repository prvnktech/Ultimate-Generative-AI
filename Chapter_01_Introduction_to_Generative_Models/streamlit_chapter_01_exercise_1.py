import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf1

# 1. Page Config
st.set_page_config(page_title="BigGAN Generator Project", layout="centered")

# 2. Compatibility Bridge
# This is crucial for Python 3.11 to handle the old BigGAN format
if not tf1.executing_eagerly():
    pass 
else:
    tf1.disable_eager_execution()

@st.cache_resource
def get_biggan_session():
    # Load using the legacy Module format explicitly
    module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-256/1')
    
    # Setup the graph inputs
    z_ph = tf1.placeholder(tf.float32, shape=[None, 128])
    y_ph = tf1.placeholder(tf.float32, shape=[None, 1000])
    tr_ph = tf1.placeholder(tf.float32)
    
    # Link inputs to the module
    out_t = module(dict(z=z_ph, y=y_ph, truncation=tr_ph))
    
    # Start the session
    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())
    return sess, z_ph, y_ph, tr_ph, out_t

# 3. UI Elements
st.title("🚗 BigGAN Vehicle Generator")
st.info("Generating images using the pretrained BigGAN-deep-256 model.")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    vehicle_type = st.selectbox("Vehicle Class", ["Sports Car", "School Bus", "Taxi"])
    v_id = {"Sports Car": 817, "School Bus": 779, "Taxi": 468}[vehicle_type]
    
    truncation = st.slider("Truncation (Variety vs. Fidelity)", 0.02, 1.0, 0.5)
    seed = st.number_input("Random Seed", value=42)

# 4. Generation Logic
if st.button("Generate Image", type="primary"):
    try:
        with st.spinner("Calculating pixels..."):
            sess, z_ph, y_ph, tr_ph, out_t = get_biggan_session()
            
            # Prepare the specific class vector
            y_val = np.zeros((1, 1000))
            y_val[0, v_id] = 1
            
            # Use seed for reproducibility
            np.random.seed(seed)
            z_val = np.random.normal(size=(1, 128))
            
            # Execute the graph
            raw_img = sess.run(out_t, feed_dict={
                z_ph: z_val, 
                y_ph: y_val, 
                tr_ph: truncation
            })
            
            # Post-process: Normalize and convert to image
            # BigGAN output is usually in range [-1, 1]
            processed = ((raw_img[0] + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            final_img = PIL.Image.fromarray(processed)
            
            # 5. Display with the updated container parameter
            st.image(final_img, caption=f"Generated {vehicle_type}", use_container_width=True)
            
            # Optional: Add a download button
            import io
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            st.download_button("Download Image", buf.getvalue(), f"{vehicle_type}.png", "image/png")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Tip: If you see a 'Module' error, try: pip install tensorflow-hub==0.15.0")