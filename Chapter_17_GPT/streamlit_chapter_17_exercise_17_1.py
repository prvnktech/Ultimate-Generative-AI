import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING (Adapted from Reference)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GPT-2 Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/docs/transformers/index',
        'About': 'Interactive GPT-2 Text Generation Lab using Hugging Face Transformers'
    }
)

# Clean, Modern CSS Design System (Reuse of reference styles)
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header - Purple/Blue Gradient for NLP/AI */
    .hero-header {
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
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

    /* Card Styling */
    .info-card {
        background-color: rgba(124, 58, 237, 0.05);
        border-left: 4px solid #7C3AED;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid #e5e7eb;
        text-align: center;
        opacity: 0.8;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE LOGIC: Hugging Face Transformers
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Load GPT-2 model and tokenizer.
    Cached to prevent reloading on every interaction.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

def generate_text(prompt_text, max_length, tokenizer, model):
    """
    Generates text using the pre-trained GPT-2 model.
    """
    # Encode input
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    
    # Generate output
    # Adding pad_token_id to avoid warnings, using eos_token_id as pad usually for GPT2
    pad_token_id = tokenizer.eos_token_id
    
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1,
        pad_token_id=pad_token_id,
        do_sample=True, # Added for more natural variation
        top_k=50
    )
    
    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🤖 GPT-2 Generation Lab</div>
    <div class="hero-subtitle">Interactive Text Completion with Hugging Face Transformers</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">📚 Hugging Face Transformers</h4>
    <p>This lab demonstrates how to use the <strong>Transformers</strong> library to access pre-trained models like <strong>GPT-2</strong>. 
    GPT-2 is a causal language model that predicts the next word in a sequence, allowing it to generate coherent text based on a prompt.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_concept, tab_code, tab_demo = st.tabs(["1. Core Concepts", "2. Implementation Code", "3. Interactive Playground"])

# ---------------------
# TAB 1: CONCEPTS
# ---------------------
with tab_concept:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 The Transformer Library")
        st.markdown("""
        Hugging Face offers a comprehensive library for accessing pre-trained models, including:
        * **GPT-2 / GPT-3**: Autoregressive models for text generation.
        * **BERT**: Bi-directional models for understanding.
        * **GPT-Neo**: Open alternatives to GPT-3.
        
        It simplifies the process of downloading weights, tokenizing text, and running inference.
        """)
        
    with col2:
        st.markdown("### ⚙️ How it works")
        st.markdown("""
        1.  **Tokenization**: Converts raw text into numerical IDs (`input_ids`) the model understands.
        2.  **Inference**: The model calculates probabilities for the next token.
        3.  **Decoding**: Converts the generated numerical IDs back into human-readable text.
        """)
        st.info("Ensure you have the library installed: `pip install transformers`")

# ---------------------
# TAB 2: CODE
# ---------------------
with tab_code:
    st.markdown("### 💻 Standard Implementation")
    st.markdown("Below is the standard pattern to load and run GPT-2 using PyTorch.")
    
    st.code('''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load Pre-trained Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Prepare Input
input_text = "Once upon a time in a distant galaxy"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 3. Generate Output
output = model.generate(
    input_ids, 
    max_length=50, 
    num_return_sequences=1
)

# 4. Decode and Print
print(tokenizer.decode(output[0], skip_special_tokens=True))
    ''', language='python')

# ---------------------
# TAB 3: DEMO
# ---------------------
with tab_demo:
    st.markdown("### 🎮 Try it yourself")
    
    # Initialize Model
    with st.spinner("Loading GPT-2 Model... (this may take a moment first time)"):
        try:
            tokenizer, model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    col_input, col_settings = st.columns([2, 1])
    
    with col_settings:
        st.markdown("**Generation Settings**")
        max_len = st.slider("Max Length", min_value=10, max_value=200, value=50, help="Total length of output text (including prompt)")
        
    with col_input:
        user_prompt = st.text_area("Input Prompt", value="Once upon a time in a distant galaxy", height=100)
        
        if st.button("✨ Generate Continuation"):
            if not user_prompt:
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Generating..."):
                    try:
                        result = generate_text(user_prompt, max_len, tokenizer, model)
                        
                        st.markdown("### Result:")
                        st.markdown(f"""
                        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 20px; border-radius: 8px; color: #166534;">
                            {result}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Generation Error: {e}")

# Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🤖 GPT-2 Lab</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem;">
            Powered by Hugging Face Transformers & PyTorch
        </p>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • Transformers
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)