"""
Gradient Descent Optimization Demo
Interactive demonstration of linear regression optimization using Gradient Descent
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Gradient Descent Lab",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://en.wikipedia.org/wiki/Gradient_descent',
        'About': 'Interactive Gradient Descent Optimization Demonstration'
    }
)

# Clean, Modern CSS Design System
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
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

    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }

    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        color: inherit;
    }

    /* Card Styling */
    .info-card, .success-card, .warning-card, .metric-card {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid rgba(128, 128, 128, 0.1);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Theme-aware card backgrounds */
    .info-card {
        background-color: rgba(59, 130, 246, 0.05);
        border-left: 4px solid #3b82f6;
    }

    .success-card {
        background-color: rgba(16, 185, 129, 0.05);
        border-left: 4px solid #10b981;
    }

    .metric-card {
        background-color: transparent;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    /* Metric Values */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4F46E5;
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
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# Core Logic: Gradient Descent
@st.cache_data
def run_gradient_descent(x, y, learning_rate, epochs):
    """Run gradient descent optimization for linear regression"""
    m, c = 0.0, 0.0  # Initial parameters (slope, intercept)
    n = len(x)
    history = {'m': [], 'c': [], 'loss': [], 'epoch': []}
    
    # Store initial state
    y_pred_init = m * x + c
    loss_init = np.mean((y - y_pred_init) ** 2)
    history['m'].append(m)
    history['c'].append(c)
    history['loss'].append(loss_init)
    history['epoch'].append(0)

    for i in range(epochs):
        y_pred = m * x + c
        
        # Calculate gradients
        dm = -(2/n) * np.sum(x * (y - y_pred))
        dc = -(2/n) * np.sum(y - y_pred)
        
        # Update parameters
        m = m - learning_rate * dm
        c = c - learning_rate * dc
        
        # Calculate loss
        loss = np.mean((y - (m * x + c)) ** 2)
        
        # Log history (downsample for performance if epochs > 200)
        if epochs <= 200 or i % (epochs // 200) == 0 or i == epochs - 1:
            history['m'].append(m)
            history['c'].append(c)
            history['loss'].append(loss)
            history['epoch'].append(i + 1)
            
    return m, c, history

def add_keyboard_shortcuts():
    """Add keyboard shortcuts for better UX"""
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+G or Cmd+G to run
        if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
            e.preventDefault();
            const generateBtn = document.querySelector('button[kind="primary"]');
            if (generateBtn) generateBtn.click();
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Add keyboard shortcuts
add_keyboard_shortcuts()

# Award-Winning Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">📉 Gradient Descent Optimization</div>
    <div class="hero-subtitle">Interactive Visualization of Linear Regression Training</div>
</div>
""", unsafe_allow_html=True)

# Main Application Logic
st.markdown("""
<div class="info-card">
    <h4 style="margin-top: 0;">🔄 Visualizing Optimization Algorithms</h4>
    <p>Experiment with Gradient Descent to optimize a linear regression model. Visualize how the algorithm iteratively finds the best-fit line by minimizing the loss function.</p>
</div>
""", unsafe_allow_html=True)

# Configuration in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚙️ Hyperparameters")
    learning_rate = st.number_input("Learning Rate (α)", 
                                  min_value=0.0001, max_value=0.1, 
                                  value=0.01, step=0.001, format="%.4f",
                                  help="Step size for each iteration")
    epochs = st.slider("Epochs (Iterations)", 
                     min_value=10, max_value=5000, 
                     value=1000, step=10,
                     help="Number of passes through the training data")

# Initialize custom data variables
x_custom, y_custom = None, None

with col2:
    st.markdown("### 📊 Data Configuration")
    data_mode = st.radio("Data Source", 
                         ["Simple Example (Fixed)", "Random Linear Data", "Custom Data (Edit/Upload)"], 
                         horizontal=True)
    
    if data_mode == "Random Linear Data":
        # Increased range for "Large Dataset" testing
        noise_level = st.slider("Noise Level", 0.0, 5.0, 1.0)
        n_points = st.slider("Number of Points", 5, 5000, 20, help="Higher values verify large dataset performance")
    elif data_mode == "Custom Data (Edit/Upload)":
        tab1, tab2 = st.tabs(["✍️ Manual Entry", "📂 Upload CSV"])
        
        with tab1:
            st.markdown("Edit the table below:")
            # Initialize default data if not present
            if 'custom_df' not in st.session_state:
                st.session_state.custom_df = pd.DataFrame({
                    'x': [1.0, 2.0, 3.0, 4.0, 5.0],
                    'y': [1.5, 3.8, 6.2, 8.5, 9.8]
                })
            
            edited_df = st.data_editor(st.session_state.custom_df, num_rows="dynamic", key="data_editor")
            # Store current edited values for potential use
            x_custom = edited_df['x'].values
            y_custom = edited_df['y'].values
            
        with tab2:
            uploaded_file = st.file_uploader("Upload CSV (must contain 'x' and 'y' columns)", type='csv')
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'x' in df.columns and 'y' in df.columns:
                        st.success(f"Data loaded successfully! ({len(df)} rows)")
                        # Override manual entry if file is uploaded
                        x_custom = df['x'].values
                        y_custom = df['y'].values
                    else:
                        st.error("CSV must have 'x' and 'y' columns.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.info("Using fixed dataset: x=[1,2,3,4,5], y=[5,7,9,11,13]")

if st.button("🚀 Run Gradient Descent", use_container_width=True, type="primary"):
    error_flag = False
    
    # Prepare Data
    if data_mode == "Simple Example (Fixed)":
        x_train = np.array([1, 2, 3, 4, 5])
        y_train = np.array([5, 7, 9, 11, 13])
    elif data_mode == "Custom Data (Edit/Upload)":
        if x_custom is not None and len(x_custom) > 1:
            # Ensure numeric types
            try:
                x_train = np.array(x_custom, dtype=float)
                y_train = np.array(y_custom, dtype=float)
            except ValueError:
                st.error("Custom data contains non-numeric values. Please check your input.")
                error_flag = True
        else:
            st.error("Please provide at least 2 data points for custom data.")
            error_flag = True
    else:
        np.random.seed(42)
        x_train = np.linspace(0, 10, n_points)
        # True relation y = 2x + 3
        y_train = 2 * x_train + 3 + np.random.normal(0, noise_level, n_points)
        
    if not error_flag:
        st.session_state['gd_run_active'] = True
        st.session_state['gd_data'] = (x_train, y_train)

if st.session_state.get('gd_run_active', False):
    x_train, y_train = st.session_state.get('gd_data', (np.array([1, 2, 3, 4, 5]), np.array([5, 7, 9, 11, 13])))
    
    # Run Optimization
    # NumPy handles large arrays efficiently, but this loop can take time for huge N * Epochs
    final_m, final_c, history = run_gradient_descent(x_train, y_train, learning_rate, epochs)
    
    st.markdown("---")
    st.markdown('<h2 class="subsection-header">📊 Optimization Results</h2>', unsafe_allow_html=True)

    # Metrics Row
    m_col, c_col, loss_col = st.columns(3)
    with m_col:
        st.metric("Final Slope (m)", f"{final_m:.4f}")
    with c_col:
        st.metric("Final Intercept (c)", f"{final_c:.4f}")
    with loss_col:
        st.metric("Final Loss (MSE)", f"{history['loss'][-1]:.4f}")

    # Visualization Columns
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Regression Fit Plot
        fig_fit = go.Figure()
        
        # PERFORMANCE OPTIMIZATION: Downsample visualization for large datasets
        # Plotting >2k-5k points in browser JS is slow/crash-prone
        MAX_VIZ_POINTS = 2000
        total_points = len(x_train)
        
        if total_points > MAX_VIZ_POINTS:
            # Randomly sample indices
            indices = np.random.choice(total_points, MAX_VIZ_POINTS, replace=False)
            x_viz = x_train[indices]
            y_viz = y_train[indices]
            viz_name = f'Training Data (Subset of {MAX_VIZ_POINTS})'
            viz_note = f"⚠️ Displaying a random subset of {MAX_VIZ_POINTS} points out of {total_points} for performance."
        else:
            x_viz = x_train
            y_viz = y_train
            viz_name = 'Training Data'
            viz_note = ""

        # Scatter plot of data points
        fig_fit.add_trace(go.Scatter(
            x=x_viz, y=y_viz,
            mode='markers',
            name=viz_name,
            marker=dict(color='#667eea', size=6, opacity=0.6) # Slightly smaller/transparent for dense data
        ))
        
        # Regression line (calculated on full range)
        x_range = np.linspace(min(x_train), max(x_train), 100)
        y_pred_line = final_m * x_range + final_c
        
        fig_fit.add_trace(go.Scatter(
            x=x_range, y=y_pred_line,
            mode='lines',
            name='Fitted Line',
            line=dict(color='#e53e3e', width=3)
        ))
        
        fig_fit.update_layout(
            title="Linear Regression Fit",
            xaxis_title="Input (x)",
            yaxis_title="Output (y)",
            height=400,
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig_fit, width='stretch')
        if viz_note:
            st.caption(viz_note)
            
    with viz_col2:
        # Loss Curve Plot
        fig_loss = go.Figure()
        
        fig_loss.add_trace(go.Scatter(
            x=history['epoch'],
            y=history['loss'],
            mode='lines',
            name='Loss',
            line=dict(color='#38a169', width=2)
        ))
        
        fig_loss.update_layout(
            title="Loss Curve (MSE over Epochs)",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error",
            height=400
        )
        st.plotly_chart(fig_loss, width='stretch')
        
    st.markdown("""
    <div class="success-card">
        <h4 style="margin: 0;">💡 How it works</h4>
        <p>The algorithm iteratively updates the slope (m) and intercept (c) by moving in the opposite direction of the gradient of the loss function.
        As training progresses (epochs increase), the regression line fits the data better, and the loss decreases, eventually stabilizing (convergence).</p>
    </div>
    """, unsafe_allow_html=True)

# Professional Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">📉 Gradient Descent Lab</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem;">
            Interactive exploration of optimization algorithms and machine learning fundamentals.
        </p>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • NumPy • Plotly
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)