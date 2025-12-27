import streamlit as st
import os
import sys
import subprocess
import time

# -----------------------------------------------------------------------------
# COMPATIBILITY FIX: Keras 3 & Transformers
# -----------------------------------------------------------------------------
# Recent versions of TensorFlow/Keras (Keras 3) require this environment variable 
# or the 'tf-keras' package to work with Hugging Face Transformers.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tensorflow as tf
    from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
except Exception as e:
    # Catch specific Keras 3 compatibility errors
    if "Keras 3" in str(e) or "tf-keras" in str(e):
        st.set_page_config(page_title="Setup Error", page_icon="⚠️")
        st.error("⚠️ **Keras 3 Compatibility Error Detected**")
        
        st.markdown(f"""
        The installed version of Keras (Keras 3) is currently incompatible with the Transformers library without an adapter.
        
        **Error Details:**
        `{e}`
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("### Option 1: Manual Install (Recommended)")
            st.markdown("Run this in your terminal:")
            st.code("pip install tf-keras")
            st.markdown("Then **stop (Ctrl+C)** and **restart** this app.")
            
        with col2:
            st.warning("### Option 2: Auto-Install")
            st.markdown("Attempt to install via script:")
            if st.button("🚀 Install tf-keras & Exit"):
                try:
                    with st.spinner("Installing tf-keras..."):
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-keras"])
                    st.success("✅ Installation successful! Please restart the app manually.")
                    time.sleep(2)
                    sys.exit() # Exit so user is forced to restart
                except Exception as install_err:
                    st.error(f"Auto-install failed: {install_err}")
                    st.error("Please try Option 1.")
        
        st.stop()
    else:
        # Re-raise other import errors (e.g., missing tensorflow)
        raise e

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TensorFlow GPT Lab",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/docs/transformers/model_doc/gpt2',
        'About': 'Requirement 17.4: Fine-tuning GPT models with TensorFlow'
    }
)

# Modern CSS Design System (TensorFlow Theme: Orange/Gold)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header - TensorFlow Theme (Orange to Yellow) */
    .hero-header {
        background: linear-gradient(135deg, #FF6F00 0%, #FFCA28 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
        color: white !important;
    }

    .info-card {
        background-color: rgba(255, 111, 0, 0.05);
        border-left: 4px solid #FF6F00;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-box {
        background-color: #fff7ed; /* orange-50 */
        border: 1px solid #fed7aa; /* orange-200 */
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        background-color: #FF6F00;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #E65100; /* Darker orange */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOGIC: Model Loading (TensorFlow)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tf_model():
    """
    Loads GPT-2 model and tokenizer using TensorFlow classes.
    """
    try:
        # Note the 'TF' prefix in the model class name
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = TFGPT2LMHeadModel.from_pretrained("gpt2")
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🌊 Requirement 17.4: TensorFlow</div>
    <div class="hero-subtitle">Fine-Tuning Concepts using TFGPT2LMHeadModel</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🧪 Deep Learning Frameworks: TensorFlow</h4>
    <p>This lab implements the fine-tuning logic using <strong>TensorFlow</strong> (Keras). 
    While the concepts (Tokenization -> Forward Pass -> Loss) are the same as PyTorch, the syntax and data structures differ slightly (e.g., <code>return_tensors="tf"</code>).</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_concepts, tab_code, tab_lab = st.tabs(["1. Core Concepts", "2. TensorFlow Code", "3. Interactive Lab"])

# --- TAB 1: CONCEPTS ---
with tab_concepts:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ TensorFlow Specifics")
        st.markdown("""
        * **`TFGPT2LMHeadModel`**: The TensorFlow version of the GPT-2 model class.
        * **`return_tensors="tf"`**: Tells the tokenizer to output TensorFlow tensors (`tf.Tensor`) instead of PyTorch tensors (`"pt"`) or Python lists.
        * **Eager Execution**: TensorFlow 2.x runs eagerly by default, allowing us to inspect tensor values immediately like in PyTorch.
        """)
    with col2:
        st.markdown("### 📉 The Forward Pass")
        st.markdown("""
        Just like in PyTorch, passing `labels` equal to `input_ids` triggers the internal loss calculation.
        
        The model shifts the labels internally so that it learns to predict the next token:
        * Input: `[A, B, C]`
        * Target: `[B, C, D]`
        """)

# --- TAB 2: CODE ---
with tab_code:
    st.markdown("### 💻 TensorFlow Implementation")
    st.markdown("Below is the corrected code snippet adapted for TensorFlow.")
    st.code('''
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# 1. Load the pre-trained GPT-2 tokenizer and model (TF version)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# 2. Encode input text into TensorFlow tensor format
input_text = "Fine-tuning GPT models"
# Note: return_tensors="tf"
input_ids = tokenizer(input_text, return_tensors="tf").input_ids

# 3. Forward pass
# TFGPT2LMHeadModel returns loss when labels are provided
outputs = model(input_ids, labels=input_ids)

# 4. Extract and print the loss value
# outputs.loss is usually a scalar tensor for the batch
loss = outputs.loss
print(f"Training loss: {float(loss):.4f}")
    ''', language='python')

# --- TAB 3: INTERACTIVE LAB ---
with tab_lab:
    st.markdown("### 🧪 Experiment with TensorFlow")
    
    # Initialize
    with st.spinner("Initializing TensorFlow Model... (This may take a moment)"):
        tokenizer, model, err = load_tf_model()
        if err:
            st.error(f"Failed to load model: {err}")
            st.stop()
        else:
            st.success("TensorFlow Model Ready!", icon="✅")

    col_input, col_viz = st.columns([1, 1])
    
    with col_input:
        st.subheader("1. Custom Dataset Input")
        user_input = st.text_area("Enter text to fine-tune on:", value="Fine-tuning GPT models", height=150)
        
        run_btn = st.button("▶️ Run Forward Pass")
        
    with col_viz:
        st.subheader("2. Training Metrics")
        
        if run_btn and user_input:
            try:
                # 1. Tokenize (TensorFlow format)
                inputs = tokenizer(user_input, return_tensors="tf")
                input_ids = inputs.input_ids
                
                # 2. Forward Pass
                outputs = model(input_ids, labels=input_ids)
                
                # 3. Get Loss
                # In TF, we often convert to numpy or float for display
                loss_val = float(outputs.loss)
                
                # Visuals
                st.markdown("**Input Tensor Shape:**")
                st.code(f"{input_ids.shape}", language="python")
                
                st.markdown("**Calculated Loss:**")
                st.markdown(f"""
                <div class="metric-box">
                    <h2 style="margin:0; color: #E65100;">{loss_val:.4f}</h2>
                    <p style="margin:0; color: #9A3412;">Cross Entropy Loss</p>
                </div>
                """, unsafe_allow_html=True)
                
                if loss_val < 3.0:
                    st.info("✅ Low Loss: predictable text.")
                else:
                    st.warning("⚠️ High Loss: surprising text.")
                    
            except Exception as e:
                st.error(f"Computation Error: {e}")
                st.info("Tip: Ensure 'tensorflow' and 'tf-keras' are installed in your environment.")
        elif run_btn:
            st.warning("Please enter text first.")

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; border-top: 1px solid #e5e7eb; text-align: center; opacity: 0.8;">
    <p><strong>Requirement 17.4 Lab (TF Edition)</strong> • Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RUNNER SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if st.runtime.exists():
        pass
    else:
        try:
            from streamlit.web import cli as stcli
        except ImportError:
            from streamlit import cli as stcli
        
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())