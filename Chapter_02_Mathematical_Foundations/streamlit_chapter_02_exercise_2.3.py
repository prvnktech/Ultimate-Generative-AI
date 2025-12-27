import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NumPy Neural Network",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://numpy.org/doc/stable/',
        'About': 'Requirement 2.3: Basic Neural Network Implementation'
    }
)

# Modern CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header - Neural Network Gradient */
    .hero-header {
        background: linear-gradient(135deg, #4F46E5 0%, #9333EA 100%);
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
        background-color: rgba(79, 70, 229, 0.05);
        border-left: 4px solid #4F46E5;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        background-color: #4F46E5;
        color: white;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        color: white;
    }
    
    .result-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE LOGIC: The Provided Code
# -----------------------------------------------------------------------------

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function"""
    return x * (1 - x)

def train_network(epochs, learning_rate, progress_bar, status_text, chart_placeholder):
    """
    Executes the training loop based on the requirement.
    Includes callbacks for UI updates.
    """
    # 1. Training data: inputs and corresponding outputs (XOR Pattern)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    # 2. Initialize weights randomly with mean 0
    np.random.seed(1)
    # Weights shape: (2 inputs -> 1 output)
    weights = 2 * np.random.random((2, 1)) - 1
    bias = np.random.random(1)

    error_history = []
    
    # Update UI less frequently to keep it fast
    update_interval = epochs // 20 if epochs >= 20 else 1

    # 3. Training loop
    for epoch in range(epochs):
        # Forward propagation
        input_layer = inputs
        # Dot product + bias
        output_layer = sigmoid(np.dot(input_layer, weights) + bias)
        
        # Calculate error
        error = outputs - output_layer
        
        # Track mean absolute error for visualization
        mean_error = np.mean(np.abs(error))
        error_history.append(mean_error)
        
        # Backpropagation
        adjustments = error * sigmoid_derivative(output_layer)
        weights += np.dot(input_layer.T, adjustments) * learning_rate
        bias += np.sum(adjustments) * learning_rate

        # Update UI
        if epoch % update_interval == 0 or epoch == epochs - 1:
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Error: {mean_error:.4f}")
            
            # Dynamic Chart (Live updates)
            if len(error_history) > 1:
                chart_df = pd.DataFrame(error_history, columns=["Mean Error"])
                chart_placeholder.line_chart(chart_df)
                
            # Simulate slight delay for visual effect only if epochs are low
            if epochs < 1000:
                time.sleep(0.001)

    return output_layer, weights, bias, error_history

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🕸️ Basic Neural Network</div>
    <div class="hero-subtitle">NumPy Implementation of Backpropagation (Requirement 2.3)</div>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🧠 Perceptron Logic</h4>
    <p>This app implements a <strong>single-layer neural network</strong> from scratch using NumPy. 
    It attempts to learn the <strong>XOR</strong> function (Exclusive OR) using Gradient Descent and Backpropagation.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_train, tab_code, tab_concept = st.tabs(["1. Interactive Training", "2. Source Code", "3. Architecture Details"])

# --- TAB 1: INTERACTIVE TRAINING ---
with tab_train:
    col_settings, col_viz = st.columns([1, 2])
    
    with col_settings:
        st.subheader("⚙️ Hyperparameters")
        
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, help="Step size for weight updates.")
        n_epochs = st.slider("Epochs", 100, 20000, 10000, 100, help="Number of training iterations.")
        
        st.markdown("### Training Data (XOR)")
        st.code("""
Input A | Input B | Target
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
        """)
        
        start_btn = st.button("🚀 Train Network", type="primary")

    with col_viz:
        st.subheader("📈 Training Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        if start_btn:
            final_output, final_weights, final_bias, history = train_network(
                n_epochs, lr, progress_bar, status_text, chart_placeholder
            )
            
            st.success("Training Complete!")
            
            # --- FINAL STATIC PLOT (ADDED) ---
            st.markdown("### 📉 Final Error Rate Graph")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history, color='#4F46E5', linewidth=2)
            ax.set_title("Training Error over Epochs")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Mean Absolute Error")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            # ---------------------------------
            
            st.markdown("### 🎯 Final Predictions")
            
            # Format results for display
            results_df = pd.DataFrame({
                "Input A": [0, 0, 1, 1],
                "Input B": [0, 1, 0, 1],
                "Target": [0, 1, 1, 0],
                "Prediction": [f"{x[0]:.4f}" for x in final_output],
                "Rounded": [int(round(x[0])) for x in final_output]
            })
            st.table(results_df)
            
            # Explanation of Single Layer Limit
            is_perfect = all(results_df['Target'] == results_df['Rounded'])
            if not is_perfect:
                st.warning("""
                **Note on XOR:** You might notice the predictions aren't perfect. 
                A single-layer perceptron (linear classifier) often struggles to solve XOR perfectly because XOR is not linearly separable. 
                To solve this perfectly, a **Hidden Layer** is usually required.
                """)
            else:
                st.success("The model managed to converge to a solution!")

# --- TAB 2: CODE ---
with tab_code:
    st.markdown("### 💻 NumPy Implementation")
    st.code('''
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Initialize weights randomly with mean 0
np.random.seed(1)
weights = 2 * np.random.random((2, 1)) - 1
bias = np.random.random(1)

# Learning rate and number of epochs
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    input_layer = inputs
    output_layer = sigmoid(np.dot(input_layer, weights) + bias)
    
    # Calculate error
    error = outputs - output_layer
    
    # Backpropagation
    adjustments = error * sigmoid_derivative(output_layer)
    weights += np.dot(input_layer.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate

# Final output after training
print(f"Trained Output: \\n{output_layer}")
    ''', language='python')

# --- TAB 3: CONCEPT ---
with tab_concept:
    st.markdown("### 🧠 How it Works")
    
    st.markdown("**1. Initialization**")
    st.markdown("Weights are initialized randomly. The neural network starts knowing nothing.")
    
    st.markdown("**2. Forward Propagation**")
    st.latex(r"Output = \sigma(Input \cdot Weights + Bias)")
    st.markdown("The inputs flow through the weights and passed through the Sigmoid activation function to squash the result between 0 and 1.")
    
    st.markdown("**3. Error Calculation**")
    st.latex(r"Error = Target - Output")
    
    st.markdown("**4. Backpropagation**")
    st.markdown("We adjust the weights based on how much they contributed to the error. We use the derivative of the sigmoid function to determine the direction and magnitude of the update.")
    st.latex(r"W_{new} = W_{old} + (Input^T \cdot (Error \cdot \sigma'(Output))) \cdot \eta")

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; border-top: 1px solid #e5e7eb; text-align: center; opacity: 0.8;">
    <p><strong>Requirement 2.3 Lab</strong> • Built with Streamlit & NumPy</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RUNNER SCRIPT
# -----------------------------------------------------------------------------
import os
import sys

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