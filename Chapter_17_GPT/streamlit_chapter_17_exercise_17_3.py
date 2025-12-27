import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import sys

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PyTorch GPT Lab",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/docs/transformers/model_doc/gpt2',
        'About': 'Interactive PyTorch & GPT-2 Implementation Lab'
    }
)

# Modern CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header - PyTorch Colors (Orange/Red) */
    .hero-header {
        background: linear-gradient(135deg, #EE4C2C 0%, #C13016 100%);
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
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        color: white !important;
    }

    .info-card {
        background-color: rgba(238, 76, 44, 0.05);
        border-left: 4px solid #EE4C2C;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .code-explanation {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Tensor Visualization */
    .tensor-box {
        font-family: 'Courier New', monospace;
        background-color: #1e293b;
        color: #a5f3fc;
        padding: 10px;
        border-radius: 6px;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOGIC: Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pytorch_model():
    """
    Loads the standard GPT-2 model and tokenizer.
    Cached to prevent reloading on every interaction.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔥 PyTorch & GPT-2 Lab</div>
    <div class="hero-subtitle">Implementing Fine-Tuning Concepts with Deep Learning Frameworks</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">Requirement 17.3: Deep Learning Implementation</h4>
    <p>Python's deep learning frameworks, like <strong>PyTorch</strong>, allow for seamless implementation of GPT models. 
    This lab demonstrates the low-level steps of calculating <strong>Training Loss</strong> by manually running a forward pass on the model.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_concept, tab_code, tab_lab = st.tabs(["1. Core Concepts", "2. PyTorch Code", "3. Interactive Lab"])

# ---------------------
# TAB 1: CONCEPTS
# ---------------------
with tab_concept:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 The Forward Pass")
        st.markdown("""
        In PyTorch, training a language model involves these steps:
        1.  **Tokenization**: Converting text strings into tensor integers.
        2.  **Model Mode**: Setting `model.train()` enables dropout and gradient tracking.
        3.  **Labels**: Passing `labels` to the model triggers the internal loss calculation.
        """)
    
    with col2:
        st.markdown("### 📉 Calculating Loss")
        st.markdown("""
        GPT-2 is trained using **Causal Language Modeling**. 
        
        When we pass `labels=input_ids`, the model automatically shifts the labels to the right. It effectively asks:
        *"Given word 1, did I predict word 2 correctly? Given word 1 and 2, did I predict word 3?"*
        
        The **Loss** represents how "surprised" the model is by the actual next word. Lower loss = better prediction.
        """)

# ---------------------
# TAB 2: CODE
# ---------------------
with tab_code:
    st.markdown("### 💻 PyTorch Implementation")
    st.markdown("This code snippet demonstrates loading a model and calculating loss on a single sentence.")
    
    st.code('''
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. Ensure model is in training mode
model.train()

# 3. Encode input text into tensor format
input_text = "Fine-tuning GPT models"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 4. Forward pass
# GPT2LMHeadModel returns loss automatically when 'labels' are provided
outputs = model(input_ids=input_ids, labels=input_ids)

# 5. Extract and print the loss value
loss = outputs.loss
print(f"Training loss: {loss.item():.4f}")
    ''', language='python')

# ---------------------
# TAB 3: INTERACTIVE LAB
# ---------------------
with tab_lab:
    st.markdown("### 🧪 Experiment with Loss Calculation")
    
    # Initialize Model
    with st.spinner("Initializing PyTorch Model..."):
        try:
            tokenizer, model = load_pytorch_model()
            st.success("PyTorch Model Loaded Successfully!", icon="✅")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    # Input Section
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("#### 1. Input Data")
        user_text = st.text_input("Enter text for training sample:", value="Fine-tuning GPT models")
        
        if st.button("▶️ Run Forward Pass", type="primary"):
            if not user_text:
                st.warning("Please enter text.")
            else:
                # Store results in session state to display in right column
                try:
                    # 1. Tokenize
                    input_tensor = tokenizer(user_text, return_tensors="pt").input_ids
                    
                    # 2. Forward Pass
                    model.train() # Explicitly set to train mode as per requirement
                    with torch.no_grad(): # We use no_grad here just for UI speed/memory, in real training we wouldn't
                        outputs = model(input_ids=input_tensor, labels=input_tensor)
                    
                    loss_val = outputs.loss.item()
                    
                    st.session_state['lab_results'] = {
                        'tensor': input_tensor,
                        'loss': loss_val,
                        'tokens': [tokenizer.decode([t]) for t in input_tensor[0]]
                    }
                except Exception as e:
                    st.error(f"Computation Error: {e}")

    with col_output:
        st.markdown("#### 2. PyTorch Internals")
        
        if 'lab_results' in st.session_state:
            res = st.session_state['lab_results']
            
            st.markdown("**A. Tensor Representation (`input_ids`)**")
            st.markdown(f"The text is converted into a tensor of shape `{tuple(res['tensor'].shape)}`:")
            st.markdown(f"""<div class="tensor-box">{res['tensor'].tolist()}</div>""", unsafe_allow_html=True)
            
            st.markdown("**B. Token Mapping**")
            st.caption("How the model sees your words:")
            st.write(res['tokens'])
            
            st.markdown("---")
            
            st.markdown("**C. Calculated Loss**")
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 1px solid #22c55e; padding: 15px; border-radius: 8px; text-align: center;">
                <span style="color: #15803d; font-size: 0.9rem;">Training Loss (Cross Entropy)</span><br>
                <span style="color: #16a34a; font-size: 2.5rem; font-weight: bold;">{res['loss']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if res['loss'] < 3.0:
                st.caption("✅ Low loss: The model finds this text predictable/familiar.")
            else:
                st.caption("⚠️ High loss: The model finds this text surprising or unusual.")

        else:
            st.info("Click 'Run Forward Pass' to see the PyTorch tensors and loss calculation.")

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; border-top: 1px solid #e5e7eb; text-align: center; opacity: 0.8;">
    <p><strong>Requirement 17.3 Lab</strong> • Built with Streamlit & PyTorch</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RUNNER SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if we are running inside Streamlit to prevent recursion
    if st.runtime.exists():
        pass
    else:
        try:
            from streamlit.web import cli as stcli
        except ImportError:
            from streamlit import cli as stcli
        
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())