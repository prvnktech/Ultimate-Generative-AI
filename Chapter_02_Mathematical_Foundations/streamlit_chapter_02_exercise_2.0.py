"""
Generative Models Sampling Demo - Professional Edition
Interactive demonstration of generative model sampling from probability distributions
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import time

# Professional Page Configuration
st.set_page_config(
    page_title="Generative Models Lab",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://en.wikipedia.org/wiki/Generative_model',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Interactive Generative Models Sampling Demonstration - Professional Edition'
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

    .warning-card {
        background-color: rgba(245, 158, 11, 0.05);
        border-left: 4px solid #f59e0b;
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

    /* Sidebar Styling */
    .sidebar-header {
        background: rgba(124, 58, 237, 0.05);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(124, 58, 237, 0.1);
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Formula Box */
    .formula-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    
    /* Dark mode adjustment for formula box */
    @media (prefers-color-scheme: dark) {
        .formula-box {
            background-color: #1e293b;
            border-color: #334155;
        }
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid #e5e7eb;
        text-align: center;
        opacity: 0.8;
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 5px;
        font-size: 0.9em;
        color: #6366f1;
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

# Utility Functions
@st.cache_data
def generate_samples(distribution, params, n_samples):
    """Generate samples from specified distribution"""
    if distribution == "Normal":
        mu, sigma = params
        return np.random.normal(mu, sigma, n_samples)
    elif distribution == "Uniform":
        low, high = params
        return np.random.uniform(low, high, n_samples)
    elif distribution == "Exponential":
        scale = params[0]
        return np.random.exponential(scale, n_samples)
    elif distribution == "Beta":
        alpha, beta = params
        return np.random.beta(alpha, beta, n_samples)
    elif distribution == "Gamma":
        shape, scale = params
        return np.random.gamma(shape, scale, n_samples)
    elif distribution == "Poisson":
        lam = params[0]
        return np.random.poisson(lam, n_samples)
    else:
        return np.random.normal(0, 1, n_samples)

@st.cache_data
def get_theoretical_pdf(distribution, params, x_range):
    """Get theoretical PDF for specified distribution"""
    if distribution == "Normal":
        mu, sigma = params
        return stats.norm.pdf(x_range, mu, sigma)
    elif distribution == "Uniform":
        low, high = params
        return stats.uniform.pdf(x_range, low, high-low)
    elif distribution == "Exponential":
        scale = params[0]
        return stats.expon.pdf(x_range, 0, scale)
    elif distribution == "Beta":
        alpha, beta = params
        return stats.beta.pdf(x_range, alpha, beta)
    elif distribution == "Gamma":
        shape, scale = params
        return stats.gamma.pdf(x_range, shape, 0, scale)
    elif distribution == "Poisson":
        # For Poisson, return PMF
        lam = params[0]
        x_int = np.arange(int(max(x_range)) + 1)
        pmf = stats.poisson.pmf(x_int, lam)
        return x_int, pmf
    return None

@st.cache_data
def calculate_statistics(samples):
    """Calculate comprehensive statistics for samples"""
    return {
        "count": len(samples),
        "mean": np.mean(samples),
        "std": np.std(samples),
        "min": np.min(samples),
        "max": np.max(samples),
        "median": np.median(samples),
        "skewness": stats.skew(samples),
        "kurtosis": stats.kurtosis(samples),
        "q25": np.percentile(samples, 25),
        "q75": np.percentile(samples, 75)
    }

def get_distribution_info(distribution, params):
    """Get mathematical formulas and theoretical properties for distributions"""
    info = {}
    
    if distribution == "Normal":
        mu, sigma = params
        info['latex'] = rf"f(x) = \frac{{1}}{{\sigma\sqrt{{2\pi}}}} e^{{-\frac{{(x-\mu)^2}}{{2\sigma^2}}}}"
        info['mean'] = f"μ = {mu}"
        info['variance'] = f"σ² = {sigma**2:.3f}"
        info['description'] = "Symmetric bell-shaped distribution. Models natural phenomena and measurement errors."
    elif distribution == "Uniform":
        low, high = params
        info['latex'] = rf"f(x) = \frac{{1}}{{b-a}} \text{{ for }} a \leq x \leq b"
        info['mean'] = f"(a+b)/2 = {(low+high)/2:.3f}"
        info['variance'] = f"(b-a)²/12 = {((high-low)**2/12):.3f}"
        info['description'] = "Equal probability across range. Models random selection."
    elif distribution == "Exponential":
        scale = params[0]
        info['latex'] = rf"f(x) = \lambda e^{{-\lambda x}} \text{{ where }} \lambda = \frac{{1}}{{\text{{scale}}}}"
        info['mean'] = f"scale = {scale}"
        info['variance'] = f"scale² = {scale**2:.3f}"
        info['description'] = "Memoryless distribution. Models waiting times between events."
    elif distribution == "Beta":
        alpha, beta_param = params
        mean = alpha / (alpha + beta_param)
        variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
        info['latex'] = rf"f(x) = \frac{{x^{{\alpha-1}}(1-x)^{{\beta-1}}}}{{B(\alpha,\beta)}} \text{{ for }} 0 \leq x \leq 1"
        info['mean'] = f"α/(α+β) = {mean:.3f}"
        info['variance'] = f"{variance:.4f}"
        info['description'] = "Flexible bounded [0,1] distribution. Models proportions and probabilities."
    elif distribution == "Gamma":
        shape, scale = params
        info['latex'] = rf"f(x) = \frac{{x^{{k-1}}e^{{-x/\theta}}}}{{\theta^k \Gamma(k)}} \text{{ for }} x > 0"
        info['mean'] = f"k×θ = {shape*scale:.3f}"
        info['variance'] = f"k×θ² = {shape*scale**2:.3f}"
        info['description'] = "Positive values with flexible shape. Models waiting times and rates."
    elif distribution == "Poisson":
        lam = params[0]
        info['latex'] = rf"P(X=k) = \frac{{\lambda^k e^{{-\lambda}}}}{{k!}}"
        info['mean'] = f"λ = {lam}"
        info['variance'] = f"λ = {lam}"
        info['description'] = "Discrete counting process. Models number of events in fixed interval."
    
    return info

def help_icon(text):
    """Create an inline help icon with tooltip"""
    return f'<span class="tooltip" title="{text}">ⓘ</span>'

def add_keyboard_shortcuts():
    """Add keyboard shortcuts for better UX"""
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+G or Cmd+G to generate samples
        if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
            e.preventDefault();
            const generateBtn = document.querySelector('button[kind="primary"]');
            if (generateBtn) generateBtn.click();
        }
        // Ctrl+H or Cmd+H to toggle help
        if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
            e.preventDefault();
            const expanders = document.querySelectorAll('[data-testid="stExpander"]');
            if (expanders[0]) expanders[0].click();
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Add keyboard shortcuts
add_keyboard_shortcuts()

# Award-Winning Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎲 Generative Models Laboratory</div>
    <div class="hero-subtitle">Interactive Exploration of Probability Sampling & Synthetic Data Generation</div>
</div>
""", unsafe_allow_html=True)

# Collapsible Introduction Section
with st.expander("📖 About Generative Sampling & Quick Start Guide", expanded=False):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 About Generative Sampling
        Generative models learn to create new data samples that resemble real data by sampling from learned probability distributions. This fundamental technique powers modern AI applications.
        
        **🔬 Sampling Theory** Drawing random samples from probability distributions to create synthetic data
        
        **🎨 Data Synthesis** Creating synthetic data for training and testing machine learning models
        
        ---
        
        ### 📚 How to Use This App
        
        1. **Select a Mode** from the sidebar:
           - 🔬 **Distribution Sampler**: Generate samples from different probability distributions
           - 🎮 **Generative Playground**: Compare multiple sample sizes and parameters
           - 🎨 **Applications Gallery**: Explore real-world use cases
        
        2. **Configure Parameters**: Use the sidebar controls to adjust distribution parameters
        
        3. **Generate & Analyze**: Click "Generate Samples" or "Run Experiment" to see results
        
        4. **Export Data**: Download generated samples for further analysis
        """)

    with col2:
        st.markdown("""
        ### 📈 Applications
        - **Data Augmentation** - Expand training datasets
        - **Monte Carlo Simulation** - Risk analysis and forecasting
        - **Synthetic Data Generation** - Privacy-preserving data
        - **Bayesian Sampling** - Posterior inference
        - **Generative AI** - Content creation
        
        ---
        
        ### ✨ Key Features
        - ✓ 6 probability distributions
        - ✓ Interactive parameter controls
        - ✓ Statistical analysis
        - ✓ Visual comparisons
        - ✓ Real-world demos
        """)

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">🧪 Experiment Selection</div>
        <p style="margin: 0; font-size: 0.9rem;">Choose your generative sampling mode</p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "",
        ["Distribution Sampler", "Generative Playground", "Applications Gallery"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick Info Panel - Make collapsible
    with st.sidebar.expander("💡 Key Concepts", expanded=False):
        st.markdown("""
        <ul style="margin: 0; padding-left: 1rem; font-size: 0.9rem; line-height: 1.7;">
            <li><b>Sampling</b> generates synthetic data</li>
            <li><b>Distributions</b> determine patterns</li>
            <li><b>Sample size</b> affects accuracy</li>
            <li><b>Parameters</b> control characteristics</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Keyboard Shortcuts - Make collapsible
    with st.sidebar.expander("⌨️ Keyboard Shortcuts", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; line-height: 1.8;">
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span><kbd style="background: white; padding: 0.25rem 0.5rem; border-radius: 6px; 
                      border: 1px solid #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #333;">Ctrl/Cmd+G</kbd></span>
                <span>Generate</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span><kbd style="background: white; padding: 0.25rem 0.5rem; border-radius: 6px; 
                      border: 1px solid #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #333;">Ctrl/Cmd+H</kbd></span>
                <span>Toggle Help</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# MODE 1: DISTRIBUTION SAMPLER
if mode == "Distribution Sampler":
    st.markdown('<h1 class="section-header">🔬 Distribution Sampler</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <h3 style="margin-top: 0;">Generative Sampling Process</h3>
        <div style="font-size: 1.2rem; margin: 1rem 0;">
            <strong>Distribution P(X) → Sample Generation → Synthetic Data</strong>
        </div>
        <p style="margin: 0;"><em>Transform probability distributions into actual data samples</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration Panel in Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Sampling Configuration")

        distribution = st.selectbox(
            "Distribution",
            ["Normal", "Uniform", "Exponential", "Beta", "Gamma", "Poisson"],
            help="Choose the probability distribution to sample from"
        )
        
        # Parameter Presets
        with st.expander("🎯 Quick Presets", expanded=False):
            if distribution == "Normal":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Standard Normal", use_container_width=True):
                        st.session_state['preset_mu'] = 0.0
                        st.session_state['preset_sigma'] = 1.0
                with col2:
                    if st.button("Wide Normal", use_container_width=True):
                        st.session_state['preset_mu'] = 0.0
                        st.session_state['preset_sigma'] = 3.0
            elif distribution == "Uniform":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("[0,1] Uniform", use_container_width=True):
                        st.session_state['preset_low'] = 0.0
                        st.session_state['preset_high'] = 1.0
                with col2:
                    if st.button("[-1,1] Uniform", use_container_width=True):
                        st.session_state['preset_low'] = -1.0
                        st.session_state['preset_high'] = 1.0
            elif distribution == "Exponential":
                if st.button("Unit Exponential (λ=1)", use_container_width=True):
                    st.session_state['preset_scale'] = 1.0

        n_samples = st.slider("Sample Size", 100, 10000, 1000, 100,
                            help="Number of samples to generate")

        # Initialize params with default values for all distributions
        params = [0.0, 1.0]  # Default to normal distribution params
        
        # Distribution-specific parameters
        if distribution == "Normal":
            mu = st.slider("Mean (μ)", -10.0, 10.0, 
                         st.session_state.get('preset_mu', 0.0), 0.1)
            sigma = st.slider("Std Dev (σ)", 0.1, 5.0, 
                            st.session_state.get('preset_sigma', 1.0), 0.1)
            params = [mu, sigma]
        elif distribution == "Uniform":
            low = st.slider("Lower Bound", -10.0, 10.0, 
                          st.session_state.get('preset_low', 0.0), 0.1)
            high = st.slider("Upper Bound", -10.0, 10.0, 
                           st.session_state.get('preset_high', 1.0), 0.1)
            params = [low, high]
        elif distribution == "Exponential":
            scale = st.slider("Scale (1/λ)", 0.1, 5.0, 
                            st.session_state.get('preset_scale', 1.0), 0.1)
            params = [scale]
        elif distribution == "Beta":
            alpha = st.slider("Alpha (α)", 0.1, 10.0, 2.0, 0.1)
            beta = st.slider("Beta (β)", 0.1, 10.0, 5.0, 0.1)
            params = [alpha, beta]
        elif distribution == "Gamma":
            shape = st.slider("Shape (k)", 0.1, 10.0, 2.0, 0.1)
            scale = st.slider("Scale (θ)", 0.1, 5.0, 1.0, 0.1)
            params = [shape, scale]
        elif distribution == "Poisson":
            lam = st.slider("Rate (λ)", 0.1, 20.0, 3.0, 0.1)
            params = [lam]

        if st.button("🎲 Generate Samples", use_container_width=True):
            st.session_state['samples_generated_active'] = True
            st.rerun()

    # Main Content Area
    st.markdown('<h2 class="subsection-header">📊 Sampling Results</h2>', unsafe_allow_html=True)

    # Check if samples have been generated
    if not st.session_state.get('samples_generated_active', False):
        st.markdown("""
        <div class="info-card">
            <h4 style="margin: 0 0 1rem 0;">🚀 Ready to Generate!</h4>
            <p>Configure your distribution parameters in the sidebar and click "Generate Samples" to create synthetic data.</p>
            <p><strong>Try the example:</strong> Normal distribution with μ=0, σ=1 (your original code)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Generate samples
        with st.spinner("Generating samples..."):
            samples = generate_samples(distribution, params, n_samples)
        
        # Get distribution information
        dist_info = get_distribution_info(distribution, params)
        
        # Collapsible Mathematical Formula Section
        with st.expander("📐 Mathematical Formula & Properties", expanded=False):
            st.markdown(f"### {distribution} Distribution")
            st.markdown(f"**Description:** {dist_info['description']}")
            st.markdown("#### Probability Density/Mass Function:")
            st.latex(dist_info['latex'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Theoretical Mean:** {dist_info['mean']}")
            with col2:
                st.markdown(f"**Theoretical Variance:** {dist_info['variance']}")

        # Statistics Overview
        stats_data = calculate_statistics(samples)

        st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">📈 Statistical Summary</h3>', unsafe_allow_html=True)

        # Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample Size", f"{stats_data['count']:,}")
        with col2:
            st.metric("Mean", f"{stats_data['mean']:.3f}")
        with col3:
            st.metric("Std Dev", f"{stats_data['std']:.3f}")
        with col4:
            st.metric("Median", f"{stats_data['median']:.3f}")
        
        # Collapsible Advanced Statistics & Goodness-of-Fit Tests
        with st.expander("📊 Advanced Statistics & Goodness-of-Fit Tests", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Distribution Shape**")
                st.write(f"Skewness: {stats_data['skewness']:.4f}")
                st.write(f"Kurtosis: {stats_data['kurtosis']:.4f}")
                st.write(f"Range: [{stats_data['min']:.3f}, {stats_data['max']:.3f}]")
            
            with col2:
                st.markdown("**Quartiles**")
                st.write(f"Q1 (25%): {stats_data['q25']:.3f}")
                st.write(f"Q2 (50%): {stats_data['median']:.3f}")
                st.write(f"Q3 (75%): {stats_data['q75']:.3f}")
                st.write(f"IQR: {stats_data['q75'] - stats_data['q25']:.3f}")
            
            with col3:
                st.markdown("**Goodness-of-Fit Test**")
                # Perform Kolmogorov-Smirnov test
                try:
                    if distribution == "Normal":
                        ks_stat, p_value = stats.kstest(samples, 'norm', args=(params[0], params[1]))
                    elif distribution == "Uniform":
                        ks_stat, p_value = stats.kstest(samples, 'uniform', args=(params[0], params[1]-params[0]))
                    elif distribution == "Exponential":
                        ks_stat, p_value = stats.kstest(samples, 'expon', args=(0, params[0]))
                    elif distribution == "Beta":
                        ks_stat, p_value = stats.kstest(samples, 'beta', args=(params[0], params[1]))
                    elif distribution == "Gamma":
                        ks_stat, p_value = stats.kstest(samples, 'gamma', args=(params[0], 0, params[1]))
                    else:
                        ks_stat, p_value = None, None
                    
                    if ks_stat is not None:
                        st.write(f"K-S Statistic: {ks_stat:.4f}")
                        st.write(f"p-value: {p_value:.4f}")
                        if p_value is not None and p_value > 0.05:
                            st.success("✓ Fits distribution")
                        else:
                            st.warning("⚠ May not fit perfectly")
                    else:
                        st.info("K-S test not applicable")
                except:
                    st.info("Could not compute test")

        # Distribution Visualization
        st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">📊 Distribution Analysis</h3>', unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            with st.container():
                st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                    <h4 style="margin: 0;">📊 Probability Distribution</h4>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">Empirical distribution of generated samples</p>
                </div>
                """, unsafe_allow_html=True)

            # Create histogram with theoretical overlay
            fig1 = go.Figure()

            # Histogram
            fig1.add_trace(go.Histogram(
                x=samples,
                nbinsx=50,
                name="Samples",
                opacity=0.7,
                marker_color='#667eea',
                histnorm='probability density'
            ))

            # Theoretical PDF (if available)
            try:
                if distribution == "Poisson":
                    result = get_theoretical_pdf(distribution, params, samples)
                    if result is not None:
                        x_theory, y_theory = result
                        fig1.add_trace(go.Scatter(
                            x=x_theory,
                            y=y_theory,
                            mode='markers+lines',
                            name='Theoretical PMF',
                            line=dict(color='#e53e3e', width=3)
                        ))
                else:
                    x_range = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 200)
                    theory_result = get_theoretical_pdf(distribution, params, x_range)
                    if theory_result is not None:
                        # Normal distributions and others return a single array for y values
                        y_theory = theory_result
                        fig1.add_trace(go.Scatter(
                            x=x_range,
                            y=y_theory,
                            mode='lines',
                            name='Theoretical PDF',
                            line=dict(color='#e53e3e', width=3)
                        ))
            except:
                pass

            fig1.update_layout(
                title={
                    'text': f"<b>{distribution} Distribution Sampling</b>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Inter, sans-serif', 'color': '#1e293b'}
                },
                xaxis_title="Value",
                yaxis_title="Density",
                showlegend=True,
                height=450,
                plot_bgcolor='rgba(248,250,252,0.5)',
                paper_bgcolor='rgba(255,255,255,0.8)',
                font={'family': 'Inter, sans-serif', 'size': 12, 'color': '#475569'},
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor="rgba(15, 23, 42, 0.95)",
                    font_size=13,
                    font_family="Inter, sans-serif",
                    font_color="white",
                    bordercolor="rgba(99, 102, 241, 0.5)"
                ),
                xaxis=dict(
                    gridcolor='rgba(226, 232, 240, 0.5)',
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    gridcolor='rgba(226, 232, 240, 0.5)',
                    showgrid=True,
                    zeroline=False
                ),
                margin=dict(l=60, r=40, t=80, b=60)
            )

            st.plotly_chart(fig1, width='stretch')

        with viz_col2:
            with st.container():
                st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                    <h4 style="margin: 0;">📈 Cumulative Distribution</h4>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">How samples accumulate across the distribution</p>
                </div>
                """, unsafe_allow_html=True)

                # Empirical CDF
                sorted_samples = np.sort(samples)
                y_vals = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(
                    x=sorted_samples,
                    y=y_vals,
                    mode='lines',
                    name='Empirical CDF',
                    line=dict(color='#38a169', width=3)
                ))

                # Add theoretical CDF for comparison (if available)
                try:
                    if distribution != "Poisson":
                        x_range = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 200)
                        # Initialize cdf_vals as empty array
                        cdf_vals = np.array([])
                        
                        if distribution == "Normal":
                            cdf_vals = stats.norm.cdf(x_range, params[0], params[1])
                        elif distribution == "Uniform":
                            cdf_vals = stats.uniform.cdf(x_range, params[0], params[1]-params[0])
                        elif distribution == "Exponential":
                            cdf_vals = stats.expon.cdf(x_range, 0, params[0])
                        elif distribution == "Beta":
                            cdf_vals = stats.beta.cdf(x_range, params[0], params[1])
                        elif distribution == "Gamma":
                            cdf_vals = stats.gamma.cdf(x_range, params[0], 0, params[1])

                        fig2.add_trace(go.Scatter(
                            x=x_range,
                            y=cdf_vals,
                            mode='lines',
                            name='Theoretical CDF',
                            line=dict(color='#e53e3e', width=2, dash='dash')
                        ))
                except:
                    pass

                fig2.update_layout(
                    title={
                        'text': "<b>Cumulative Distribution Function</b>",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'family': 'Inter, sans-serif', 'color': '#1e293b'}
                    },
                    xaxis_title="Value",
                    yaxis_title="Cumulative Probability",
                    showlegend=True,
                    height=450,
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    paper_bgcolor='rgba(255,255,255,0.8)',
                    font={'family': 'Inter, sans-serif', 'size': 12, 'color': '#475569'},
                    hovermode='x unified',
                    hoverlabel=dict(
                        bgcolor="rgba(15, 23, 42, 0.95)",
                        font_size=13,
                        font_family="Inter, sans-serif",
                        font_color="white",
                        bordercolor="rgba(34, 197, 94, 0.5)"
                    ),
                    xaxis=dict(
                        gridcolor='rgba(226, 232, 240, 0.5)',
                        showgrid=True,
                        zeroline=False
                    ),
                    yaxis=dict(
                        gridcolor='rgba(226, 232, 240, 0.5)',
                        showgrid=True,
                        zeroline=False,
                        range=[0, 1.05]
                    ),
                    margin=dict(l=60, r=40, t=80, b=60)
                )

                st.plotly_chart(fig2, width='stretch')

        # Sample Data Display (Collapsible)
        with st.expander("📋 Sample Data Preview & Export", expanded=False):
            # Show first 20 samples (or all if less than 20)
            n_display = min(20, len(samples))
            sample_df = pd.DataFrame({
                'Sample Index': range(1, n_display + 1),
                'Value': samples[:n_display]
            })
            
            st.markdown(f"**Showing first {n_display} of {len(samples)} samples**")
            st.dataframe(sample_df, width='stretch', height=300)

            # Download option
            full_df = pd.DataFrame({
                'Sample Index': range(1, len(samples) + 1),
                'Value': samples
            })
            csv = full_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Samples (CSV)",
                data=csv,
                file_name=f"{distribution.lower()}_samples_{n_samples}.csv",
                mime="text/csv",
                use_container_width=True
            )

# MODE 2: GENERATIVE PLAYGROUND
elif mode == "Generative Playground":
    st.markdown('<h1 class="section-header">🎮 Generative Playground</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4 style="margin-top: 0;">🧪 Experiment with Generative Parameters</h4>
        <p>Explore how different parameters and sample sizes affect the characteristics of generated data. This interactive playground helps you understand the relationship between probability distributions and synthetic data generation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration
    col1, col2 = st.columns([1, 1])

    with col1:
        playground_dist = st.selectbox(
            "Distribution Type",
            ["Normal", "Uniform", "Exponential", "Beta"],
            key="playground_dist"
        )

        sample_sizes = st.multiselect(
            "Sample Sizes to Compare",
            [100, 500, 1000, 5000, 10000],
            default=[100, 1000, 5000],
            help="Compare how sample size affects distribution estimates"
        )

    with col2:
        # Initialize all possible playground parameters with default values
        pg_mu = 0.0
        pg_sigma = 1.0
        pg_low = -1.0
        pg_high = 1.0
        pg_scale = 1.0
        pg_alpha = 2.0
        pg_beta = 2.0
        
        # Parameter controls based on distribution
        if playground_dist == "Normal":
            pg_mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, key="pg_mu")
            pg_sigma = st.slider("Std Dev (σ)", 0.1, 3.0, 1.0, key="pg_sigma")
        elif playground_dist == "Uniform":
            pg_low = st.slider("Min Value", -5.0, 0.0, -1.0, key="pg_low")
            pg_high = st.slider("Max Value", 0.0, 5.0, 1.0, key="pg_high")
        elif playground_dist == "Exponential":
            pg_scale = st.slider("Scale", 0.1, 3.0, 1.0, key="pg_scale")
        elif playground_dist == "Beta":
            pg_alpha = st.slider("Alpha", 0.1, 5.0, 2.0, key="pg_alpha")
            pg_beta = st.slider("Beta", 0.1, 5.0, 2.0, key="pg_beta")

    if st.button("🚀 Run Experiment", use_container_width=True):
        st.session_state['playground_run_active'] = True
        st.rerun()

    if st.session_state.get('playground_run_active', False):
        st.markdown('<h2 class="subsection-header">📊 Experimental Results</h2>', unsafe_allow_html=True)

        # Generate samples for different sizes
        results = {}
        for size in sample_sizes:
            if playground_dist == "Normal":
                samples = np.random.normal(pg_mu, pg_sigma, size)
            elif playground_dist == "Uniform":
                samples = np.random.uniform(pg_low, pg_high, size)
            elif playground_dist == "Exponential":
                samples = np.random.exponential(pg_scale, size)
            elif playground_dist == "Beta":
                samples = np.random.beta(pg_alpha, pg_beta, size)
            else:
                # Default to normal distribution if somehow no valid option was selected
                samples = np.random.normal(0, 1, size)

            results[size] = {
                'samples': samples,
                'stats': calculate_statistics(samples)
            }

        # Comparative Statistics
        st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">📈 Statistical Comparison</h3>', unsafe_allow_html=True)

        comparison_data = []
        for size, data in results.items():
            stats = data['stats']
            comparison_data.append({
                'Sample Size': size,
                'Mean': f"{stats['mean']:.3f}",
                'Std Dev': f"{stats['std']:.3f}",
                'Min': f"{stats['min']:.3f}",
                'Max': f"{stats['max']:.3f}",
                'Skewness': f"{stats['skewness']:.3f}"
            })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, width='stretch')

        # Visual Comparison
        st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">📊 Distribution Comparison</h3>', unsafe_allow_html=True)

        fig = go.Figure()

        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']

        for i, (size, data) in enumerate(results.items()):
            fig.add_trace(go.Histogram(
                x=data['samples'],
                nbinsx=30,
                name=f'n={size}',
                opacity=0.7,
                marker_color=colors[i % len(colors)],
                histnorm='probability density'
            ))

        fig.update_layout(
            title=f"{playground_dist} Distribution: Sample Size Comparison",
            xaxis_title="Value",
            yaxis_title="Density",
            barmode='overlay',
            height=500,
            plot_bgcolor='rgba(248,250,252,0.8)',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig, width='stretch')

        # Convergence Analysis
        st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">🎯 Convergence Analysis</h3>', unsafe_allow_html=True)

        sizes = sorted(sample_sizes)
        means = [results[size]['stats']['mean'] for size in sizes]
        stds = [results[size]['stats']['std'] for size in sizes]

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=['Mean Convergence', 'Std Dev Convergence'])

        fig2.add_trace(
            go.Scatter(x=sizes, y=means, mode='lines+markers', name='Sample Mean',
                      line=dict(color='#667eea', width=3)),
            row=1, col=1
        )

        fig2.add_trace(
            go.Scatter(x=sizes, y=stds, mode='lines+markers', name='Sample Std',
                      line=dict(color='#764ba2', width=3)),
            row=1, col=2
        )

        # Add theoretical lines
        if playground_dist == "Normal":
            fig2.add_hline(y=pg_mu, line_dash="dash", line_color="red", 
                           annotation_text=f"Theoretical Mean: {pg_mu}", row="1", col="1")
            fig2.add_hline(y=pg_sigma, line_dash="dash", line_color="red", 
                           annotation_text=f"Theoretical Std: {pg_sigma}", row="1", col="2")

        fig2.update_layout(height=400, showlegend=False)
        fig2.update_xaxes(title_text="Sample Size", row=1, col=1)
        fig2.update_xaxes(title_text="Sample Size", row=1, col=2)
        fig2.update_yaxes(title_text="Mean Value", row=1, col=1)
        fig2.update_yaxes(title_text="Standard Deviation", row=1, col=2)

        st.plotly_chart(fig2, width='stretch')

        st.markdown("""
        <div class="success-card">
            <h4 style="margin: 0 0 0.5rem 0;">💡 Key Insights</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li><b>Larger sample sizes</b> provide more accurate statistical estimates</li>
                <li><b>Means converge</b> to theoretical values as n increases (Law of Large Numbers)</li>
                <li><b>Standard deviations</b> become more stable with larger samples</li>
                <li><b>Visual distributions</b> become smoother and more defined with more data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# MODE 3: APPLICATIONS GALLERY
elif mode == "Applications Gallery":
    st.markdown('<h1 class="section-header">🎨 Applications Gallery</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4 style="margin-top: 0;">🚀 Real-World Applications of Generative Sampling</h4>
        <p>Explore how generative models and sampling techniques are used in practice across different domains and industries.</p>
    </div>
    """, unsafe_allow_html=True)

    # Application tabs
    tabs = st.tabs(["📈 Data Augmentation", "🎲 Monte Carlo Simulation", "🤖 Synthetic Data Generation", "🔬 Bayesian Sampling"])

    with tabs[0]:  # Data Augmentation
        st.markdown('<h2 class="subsection-header">📈 Data Augmentation</h2>', unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size: 1rem; margin-bottom: 1rem;">
        <strong>Data augmentation</strong> uses generative sampling to create additional training data, helping machine learning models generalize better and reduce overfitting.
        </p>
        """, unsafe_allow_html=True)

        # Example: Small dataset augmentation
        if st.button("🔄 Run Data Augmentation Demo"):
            st.session_state['aug_demo_active'] = True
            st.rerun()

        if st.session_state.get('aug_demo_active', False):
            with st.spinner("Generating augmented dataset..."):
                # Original small dataset
                np.random.seed(42)  # For reproducibility
                original_size = 100
                original_data = np.random.normal(10, 2, original_size)

                # Augment with generative sampling
                augmented_size = 500
                augmented_data = np.concatenate([
                    original_data,
                    np.random.normal(np.mean(original_data), np.std(original_data), augmented_size - original_size)
                ])

            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=original_data,
                    nbinsx=20,
                    name="Original",
                    opacity=0.7,
                    marker_color='#667eea'
                ))
                fig1.update_layout(
                    title=f"Original Dataset (n={original_size})",
                    xaxis_title="Value",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig1, width='stretch')

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=augmented_data,
                    nbinsx=30,
                    name="Augmented",
                    opacity=0.7,
                    marker_color='#38a169'
                ))
                fig2.update_layout(
                    title=f"Augmented Dataset (n={augmented_size})",
                    xaxis_title="Value",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig2, width='stretch')

            st.markdown("""
            <div class="success-card">
                <h4 style="margin: 0;">📊 Results Summary</h4>
                <ul style="margin: 0.5rem 0 0 0; padding-left: 1rem;">
                    <li>Original dataset: <strong>100 samples</strong></li>
                    <li>Augmented dataset: <strong>500 samples</strong> (5x increase)</li>
                    <li>Statistical properties preserved through generative sampling</li>
                    <li>Model training can now use much more diverse data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tabs[1]:  # Monte Carlo Simulation
        st.markdown('<h2 class="subsection-header">🎲 Monte Carlo Simulation</h2>', unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size: 1rem; margin-bottom: 1rem;">
        <strong>Monte Carlo simulation</strong> uses generative sampling to model uncertainty and risk in complex systems by running thousands of random scenarios.
        </p>
        """, unsafe_allow_html=True)

        # Investment portfolio simulation
        st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">💰 Investment Portfolio Simulation</h3>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            initial_investment = st.number_input("Initial Investment ($)", 1000, 100000, 10000)
        with col2:
            years = st.slider("Investment Horizon (years)", 1, 30, 10)
        with col3:
            n_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)

        if st.button("🎯 Run Monte Carlo Simulation"):
            st.session_state['mc_demo_active'] = True
            st.rerun()

        if st.session_state.get('mc_demo_active', False):
            with st.spinner("Running Monte Carlo simulation..."):
                # Simulate investment returns
                annual_return = 0.08  # 8% expected return
                volatility = 0.15     # 15% volatility

                # Generate random returns for each simulation
                simulated_returns = np.random.normal(
                    annual_return, volatility,
                    (n_simulations, years)
                )

                # Calculate final portfolio values
                final_values = initial_investment * np.prod(1 + simulated_returns, axis=1)

            # Results visualization
            fig = go.Figure()

            # Histogram of outcomes
            fig.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                name="Final Values",
                opacity=0.7,
                marker_color='#667eea',
                histnorm='probability'
            ))

            # Add mean line
            fig.add_vline(
                x=np.mean(final_values),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: ${np.mean(final_values):,.0f}",
                annotation_position="top"
            )

            fig.update_layout(
                title="Monte Carlo: Investment Portfolio Outcomes",
                xaxis_title="Final Portfolio Value ($)",
                yaxis_title="Probability",
                height=400
            )

            st.plotly_chart(fig, width='stretch')

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected Value", f"${np.mean(final_values):,.0f}")
            with col2:
                st.metric("95% Confidence Range", 
                         f"${np.percentile(final_values, 2.5):,.0f} - ${np.percentile(final_values, 97.5):,.0f}")
            with col3:
                st.metric("Probability of Profit", f"{(final_values > initial_investment).mean():.1%}")
            with col4:
                st.metric("Worst Case", f"${np.min(final_values):,.0f}")

    with tabs[2]:  # Synthetic Data Generation
        st.markdown('<h2 class="subsection-header">🤖 Synthetic Data Generation</h2>', unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size: 1rem; margin-bottom: 1rem;">
        <strong>Synthetic data generation</strong> creates artificial datasets that mimic real data characteristics, useful for testing, privacy-preserving data sharing, and model development.
        </p>
        """, unsafe_allow_html=True)

        # Customer data generation example
        if st.button("👥 Generate Synthetic Customer Data"):
            st.session_state['synth_demo_active'] = True
            st.rerun()

        if st.session_state.get('synth_demo_active', False):
            with st.spinner("Generating synthetic customer profiles..."):
                np.random.seed(123)
                n_customers = 1000

                # Generate realistic synthetic customer data
                ages = np.random.normal(35, 10, n_customers).clip(18, 80)
                incomes = np.random.lognormal(10.5, 0.8, n_customers)
                spending_scores = np.random.beta(2, 3, n_customers) * 100
                segments = np.random.choice(
                    ['Low Value', 'Medium Value', 'High Value'],
                    n_customers,
                    p=[0.3, 0.5, 0.2]
                )

                synthetic_df = pd.DataFrame({
                    'Customer ID': range(1, n_customers + 1),
                    'Age': ages.astype(int),
                    'Income': incomes.astype(int),
                    'Spending Score': spending_scores.astype(int),
                    'Segment': segments
                })

            # Display sample
            st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">📋 Sample Synthetic Customer Data</h3>', unsafe_allow_html=True)
            st.dataframe(synthetic_df.head(20), width='stretch')

            # Statistics
            st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">📊 Data Statistics</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Customers", f"{n_customers:,}")
                st.metric("Average Age", f"{synthetic_df['Age'].mean():.1f} years")
            
            with col2:
                st.metric("Average Income", f"${synthetic_df['Income'].mean():,.0f}")
                st.metric("Avg Spending Score", f"{synthetic_df['Spending Score'].mean():.1f}")
            
            with col3:
                segment_counts = synthetic_df['Segment'].value_counts()
                st.metric("High Value Customers", f"{segment_counts.get('High Value', 0)}")
                st.metric("Data Privacy", "✅ Protected")

            # Visualizations
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Age distribution
            axes[0].hist(synthetic_df['Age'], bins=20, alpha=0.7, color='#667eea', edgecolor='black')
            axes[0].set_title('Age Distribution')
            axes[0].set_xlabel('Age')
            axes[0].set_ylabel('Count')
            
            # Income distribution
            axes[1].hist(synthetic_df['Income'], bins=20, alpha=0.7, color='#38a169', edgecolor='black')
            axes[1].set_title('Income Distribution')
            axes[1].set_xlabel('Income ($)')
            
            # Spending score distribution
            axes[2].hist(synthetic_df['Spending Score'], bins=20, alpha=0.7, color='#d69e2e', edgecolor='black')
            axes[2].set_title('Spending Score Distribution')
            axes[2].set_xlabel('Spending Score')
            
            plt.tight_layout()
            st.pyplot(fig)

    with tabs[3]:  # Bayesian Sampling
        st.markdown('<h2 class="subsection-header">🔬 Bayesian Sampling</h2>', unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size: 1rem; margin-bottom: 1rem;">
        <strong>Bayesian sampling</strong> uses generative techniques to draw samples from posterior distributions, enabling probabilistic inference and uncertainty quantification.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4 style="margin: 0;">🎯 Bayesian Inference Example</h4>
            <p>Suppose we want to estimate the probability of success (θ) for a coin, given that we observed 7 heads in 10 flips. Using Bayesian sampling, we can generate samples from the posterior distribution.</p>
        </div>
        """, unsafe_allow_html=True)

        # Bayesian coin flip example
        if st.button("🪙 Run Bayesian Coin Inference"):
            st.session_state['bayes_demo_active'] = True
            st.rerun()

        if st.session_state.get('bayes_demo_active', False):
            with st.spinner("Sampling from posterior distribution..."):
                # Beta posterior for coin flips (7 heads, 3 tails, Beta(1,1) prior)
                n_samples = 5000
                posterior_samples = np.random.beta(7 + 1, 3 + 1, n_samples)

            col1, col2 = st.columns(2)

            with col1:
                # Posterior distribution
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=posterior_samples,
                    nbinsx=50,
                    name="Posterior",
                    opacity=0.7,
                    marker_color='#667eea',
                    histnorm='probability density'
                ))
                
                # Add theoretical Beta PDF
                x_range = np.linspace(0, 1, 200)
                beta_pdf = stats.beta.pdf(x_range, 8, 4)
                fig1.add_trace(go.Scatter(
                    x=x_range,
                    y=beta_pdf,
                    mode='lines',
                    name='Beta(8,4) PDF',
                    line=dict(color='#e53e3e', width=3)
                ))

                fig1.update_layout(
                    title="Posterior Distribution: P(θ|7H,3T)",
                    xaxis_title="Probability of Heads (θ)",
                    yaxis_title="Density",
                    height=400
                )
                st.plotly_chart(fig1, width='stretch')

            with col2:
                # Credible interval
                lower_95 = np.percentile(posterior_samples, 2.5)
                upper_95 = np.percentile(posterior_samples, 97.5)
                mean_posterior = np.mean(posterior_samples)

                st.markdown(f"""
                ### 📊 Inference Results
                
                **Data:** 7 heads, 3 tails (10 coin flips)
                
                **Posterior Summary:**
                - Mean: **{mean_posterior:.3f}**
                - 95% Credible Interval: **[{lower_95:.3f}, {upper_95:.3f}]**
                - Probability θ > 0.5: **{(posterior_samples > 0.5).mean():.1%}**
                
                **Interpretation:**
                Given the data, we're 95% confident that the true probability 
                of heads lies between {lower_95:.3f} and {upper_95:.3f}.
                """)

# Professional Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🎲 Generative Models Laboratory</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem;">
            Interactive exploration of generative sampling and synthetic data generation techniques.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🔬</div>
                <div style="font-weight: 600;">Sampling Theory</div>
                <div style="font-size: 0.9rem;">Probability distributions</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🎨</div>
                <div style="font-weight: 600;">Data Synthesis</div>
                <div style="font-size: 0.9rem;">Synthetic data generation</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">📊</div>
                <div style="font-weight: 600;">Applications</div>
                <div style="font-size: 0.9rem;">Real-world use cases</div>
            </div>
        </div>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • Plotly • NumPy • SciPy
            </p>
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Educational Purpose:</strong> Understanding generative models and sampling techniques
            </p>
            <p class="footer-text" style="margin: 0.5rem 0; font-style: italic;">
                💡 <strong>Tip:</strong> Experiment with different distributions and parameters to understand how generative sampling creates diverse synthetic data!
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)