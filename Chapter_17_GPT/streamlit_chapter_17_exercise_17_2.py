import streamlit as st
import time
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GPT Fine-Tuning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/docs/transformers/training',
        'About': 'Interactive Guide to Fine-Tuning GPT Models'
    }
)

# Shared CSS Design System (Consistent with previous labs)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header - Orange/Red Gradient for Training/Heat */
    .hero-header {
        background: linear-gradient(135deg, #EA580C 0%, #DB2777 100%);
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
        background-color: rgba(234, 88, 12, 0.05);
        border-left: 4px solid #EA580C;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .step-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOGIC: Simulation Helpers
# -----------------------------------------------------------------------------

def simulate_training_loss(epochs, steps_per_epoch):
    """
    Generates a simulated loss curve that decays over time.
    """
    total_steps = epochs * steps_per_epoch
    # Create a decaying curve with some noise
    x = np.linspace(0, total_steps, total_steps)
    # Exponential decay formula
    decay = 2.5 * np.exp(-0.05 * x) 
    # Add noise
    noise = np.random.normal(0, 0.1, total_steps)
    loss = decay + noise
    # Clip to ensure valid loss values
    loss = np.clip(loss, 0.1, 5.0)
    return loss

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔧 GPT Fine-Tuning Lab</div>
    <div class="hero-subtitle">Specializing Pre-trained Models for Domain-Specific Tasks</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🎯 Why Fine-Tune?</h4>
    <p>While pre-trained models (like GPT-2) understand general language, <strong>fine-tuning</strong> allows them to specialize. 
    By training on a specific dataset (e.g., medical records, legal documents, or movie reviews), the model learns the specific nuances, vocabulary, and context of that domain.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_guide, tab_code, tab_sim = st.tabs(["1. Step-by-Step Guide", "2. Implementation Code", "3. Training Simulator"])

# ---------------------
# TAB 1: GUIDE
# ---------------------
with tab_guide:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 1. Dataset Preparation")
        st.markdown("""
        Before training, data must be formatted correctly.
        * **Format**: JSON, CSV, or Plain Text.
        * **Library**: Hugging Face `datasets` makes loading efficient.
        * **Tokenization**: Text must be converted to numbers (Input IDs) before feeding into the model.
        """)
        st.info("Example: Loading the IMDB sentiment dataset.")

    with col2:
        st.markdown("### 🚂 2. The Training Loop")
        st.markdown("""
        Hugging Face provides a `Trainer` API that abstracts away complex loops.
        * **Hyperparameters**: Settings like `batch_size`, `learning_rate`, and `epochs`.
        * **Loss Function**: Measures how 'wrong' the model is. We want this to go down.
        * **Checkpoints**: Saving the model at intervals.
        """)

# ---------------------
# TAB 2: CODE
# ---------------------
with tab_code:
    st.markdown("### 💻 Code Walkthrough")
    st.markdown("Below is the standard pipeline to fine-tune GPT-2 using Hugging Face libraries.")
    
    st.markdown("#### Step 1: Prepare Data")
    st.code('''
from datasets import load_dataset

# Load a dataset (e.g., IMDB movie reviews)
dataset = load_dataset("imdb")

# View a sample
print(dataset["train"][0])
    ''', language='python')
    
    st.markdown("#### Step 2: Fine-Tuning Script")
    st.code('''
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. Load Pre-trained Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT-2 doesn't have a pad token by default, so we add one
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,              # Number of passes through the data
    per_device_train_batch_size=8,   # Batch size per GPU/CPU
    save_steps=500,                  # Save model every 500 steps
    logging_dir="./logs",
)

# 4. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# 5. Start Training
trainer.train()
    ''', language='python')

# ---------------------
# TAB 3: SIMULATOR
# ---------------------
with tab_sim:
    st.markdown("### 🎛️ Training Simulator")
    st.warning("⚠️ **Note:** Real fine-tuning requires significant GPU resources and time. This simulation visualizes the process.")
    
    col_config, col_viz = st.columns([1, 2])
    
    with col_config:
        st.markdown("#### Hyperparameters")
        
        sim_epochs = st.slider("Epochs", 1, 10, 3, help="How many times the model sees the entire dataset.")
        sim_lr = st.select_slider("Learning Rate", options=["1e-5", "2e-5", "5e-5", "1e-4"], value="5e-5")
        sim_batch = st.select_slider("Batch Size", options=[4, 8, 16, 32], value=8)
        
        st.markdown("---")
        start_btn = st.button("🚀 Start Mock Training", type="primary")
        
    with col_viz:
        st.markdown("#### Live Metrics")
        
        # Placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_place = st.empty()
        log_place = st.empty()
        
        if start_btn:
            status_text.markdown("**Status:** 🚀 Initializing Trainer...")
            time.sleep(1)
            
            # Simulation Data
            steps_per_epoch = 10
            total_steps = sim_epochs * steps_per_epoch
            loss_data = simulate_training_loss(sim_epochs, steps_per_epoch)
            
            current_loss = []
            
            for step in range(total_steps):
                # Update progress
                prog = (step + 1) / total_steps
                progress_bar.progress(prog)
                
                # Update status
                current_epoch = (step // steps_per_epoch) + 1
                status_text.markdown(f"**Status:** Training... Epoch {current_epoch}/{sim_epochs} | Step {step+1}")
                
                # Update Chart
                current_loss.append(loss_data[step])
                chart_data = pd.DataFrame(current_loss, columns=["Training Loss"])
                chart_place.line_chart(chart_data)
                
                # Update Logs (Last 3 lines)
                log_msg = f"Step {step+1}: Loss = {loss_data[step]:.4f}"
                log_place.code(log_msg)
                
                # Simulate processing time
                time.sleep(0.1)
            
            status_text.success("✅ Training Complete! Model saved to ./results")
            st.balloons()

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; border-top: 1px solid #e5e7eb; text-align: center; opacity: 0.8;">
    <p><strong>Requirement 17.2 Lab</strong> • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RUNNER SCRIPT
# -----------------------------------------------------------------------------
import os
import sys

if __name__ == "__main__":
    # Check if we are running inside Streamlit to prevent infinite recursion
    # or "Runtime instance already exists" errors when running 'streamlit run ...'
    if st.runtime.exists():
        pass
    else:
        try:
            from streamlit.web import cli as stcli
        except ImportError:
            from streamlit import cli as stcli
        
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())