"""
Information Theory Interactive Demo - Professional Edition
A sophisticated web application demonstrating core concepts of Information Theory
with elegant visualizations and professional design.
"""

import streamlit as st
import numpy as np
from scipy.stats import entropy
from scipy.special import rel_entr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Professional Page Configuration
st.set_page_config(
    page_title="Information Theory Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/topics/information-theory',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Interactive Information Theory Demonstration - Professional Edition'
    }
)

# Professional CSS with Modern Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Ensure good contrast for Streamlit elements */
    .stMarkdown p, .stText p, .stHeader h1, .stHeader h2, .stHeader h3 {
        color: #2d3748 !important;
    }

    /* Professional Header */
    .hero-header {
        background: linear-gradient(135deg, #4c51bf 0%, #553c9a 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(76, 81, 191, 0.4);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 0;
        color: #f7fafc;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a365d;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e2e8f0;
    }

    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 1rem 0;
    }

    /* Professional Cards */
    .metric-card {
        background: white;
        color: #1a365d;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .info-card {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        color: #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #3182ce;
        margin: 1rem 0;
    }

    .warning-card {
        background: linear-gradient(135deg, #fefcbf 0%, #faf089 100%);
        color: #744210;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #d69e2e;
        margin: 1rem 0;
    }

    .success-card {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        color: #22543d;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #38a169;
        margin: 1rem 0;
    }

    /* Enhanced Metric Display */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a365d !important;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #718096 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Professional Sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        color: #2d3748;
    }

    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Slider Styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Formula Display */
    .formula-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        font-family: 'Computer Modern', 'Latin Modern Roman', serif;
        color: #2d3748;
    }

    /* Footer */
    .footer {
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        padding: 2rem 0;
        text-align: center;
        margin-top: 3rem;
        border-radius: 8px;
        color: #4a5568;
    }

    .footer-text {
        color: #4a5568;
        font-size: 0.9rem;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def calculate_entropy(prob_dist, base=2):
    """Calculate Shannon entropy"""
    # Filter out zero probabilities
    prob_filtered = [p for p in prob_dist if p > 0]
    if len(prob_filtered) == 0:
        return 0.0
    return entropy(prob_filtered, base=base)

@st.cache_data
def calculate_kl_divergence(p, q):
    """Calculate KL divergence D(P||Q)"""
    epsilon = 1e-10
    p_safe = [pi + epsilon for pi in p]
    q_safe = [qi + epsilon for qi in q]
    return np.sum(rel_entr(p_safe, q_safe))

def normalize_distribution(values):
    """Normalize values to sum to 1"""
    total = sum(values)
    if total > 0:
        return [v / total for v in values]
    return values

def get_entropy_interpretation(h, max_h):
    """Get human-readable interpretation of entropy"""
    if max_h == 0:
        return "Undefined", "gray"
    ratio = h / max_h
    if ratio > 0.9:
        return "Very High - Nearly uniform distribution", "#27ae60"
    elif ratio > 0.7:
        return "High - Considerable uncertainty", "#2ecc71"
    elif ratio > 0.5:
        return "Medium - Moderate uncertainty", "#f39c12"
    elif ratio > 0.3:
        return "Low - Some concentration", "#e67e22"
    else:
        return "Very Low - Highly concentrated", "#e74c3c"

def get_kl_interpretation(kl):
    """Get human-readable interpretation of KL divergence"""
    if kl < 0.01:
        return "Very Low - Distributions are nearly identical", "#27ae60"
    elif kl < 0.1:
        return "Low - Distributions are similar", "#2ecc71"
    elif kl < 0.5:
        return "Medium - Noticeable differences", "#f39c12"
    elif kl < 1.0:
        return "High - Significant differences", "#e67e22"
    else:
        return "Very High - Distributions are very different", "#e74c3c"

# Professional Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔬 Information Theory Laboratory</div>
    <div class="hero-subtitle">Interactive Exploration of Entropy & KL Divergence</div>
</div>
""", unsafe_allow_html=True)

# Introduction Section
with st.container():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="margin-top: 0; color: #2d3748;">🎯 About This Laboratory</h4>
            <p style="margin-bottom: 1rem;">Welcome to the Information Theory Laboratory, an interactive platform designed for exploring fundamental concepts in information science.</p>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <h5 style="color: #3182ce; margin: 0.5rem 0;">📊 Entropy Analysis</h5>
                    <p style="font-size: 0.9rem; margin: 0;">Measure uncertainty and randomness in probability distributions</p>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h5 style="color: #3182ce; margin: 0.5rem 0;">🔄 KL Divergence</h5>
                    <p style="font-size: 0.9rem; margin: 0;">Compare how distributions differ from each other</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin-top: 0; color: #2d3748;">📈 Key Concepts</h4>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li><b>Shannon Entropy:</b> Information content measure</li>
                <li><b>KL Divergence:</b> Distribution dissimilarity</li>
                <li><b>Information Theory:</b> Foundation of data compression</li>
                <li><b>Probabilistic Modeling:</b> Machine learning basis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">🧪 Experiment Selection</div>
        <p style="margin: 0; font-size: 0.9rem; color: #718096;">Choose your analysis mode</p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "",
        ["Entropy Calculator", "KL Divergence Calculator", "Comparison Examples"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick Info Panel
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
        <h5 style="margin: 0 0 0.5rem 0; color: #2d3748;">💡 Pro Tips</h5>
        <ul style="margin: 0; padding-left: 1rem; font-size: 0.85rem; color: #718096;">
            <li>Auto-normalization ensures valid probabilities</li>
            <li>Try different distribution shapes</li>
            <li>Compare uniform vs concentrated distributions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# MODE 1: ENTROPY CALCULATOR
if mode == "Entropy Calculator":
    st.markdown('<h1 class="section-header">🎲 Entropy Analysis</h1>', unsafe_allow_html=True)

    # Mathematical Foundation
    st.markdown("""
    <div class="formula-box">
        <h3 style="margin-top: 0; color: #2d3748;">Shannon Entropy Formula</h3>
        <div style="font-size: 1.5rem; margin: 1rem 0;">
            $$H(X) = -\\sum_{i=1}^{n} p(x_i) \\log_2 p(x_i)$$
        </div>
        <p style="margin: 0; color: #718096;"><em>Measures the average information content (uncertainty) in bits</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration Panel in Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        col1, col2 = st.columns(2)
        with col1:
            n_outcomes = st.slider("Outcomes", 2, 8, 3, help="Number of possible outcomes")
        with col2:
            auto_normalize = st.checkbox("Auto-norm", True, help="Normalize probabilities to sum to 1")

        st.markdown("### 🎯 Quick Experiments")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            if st.button("🎲 Uniform", use_container_width=True):
                st.session_state.probs = [1.0 / n_outcomes] * n_outcomes
                st.rerun()
            if st.button("⭐ Peaked", use_container_width=True):
                probs = [0.1] * n_outcomes
                probs[0] = 0.9 - (n_outcomes - 1) * 0.1
                st.session_state.probs = probs
                st.rerun()

        with exp_col2:
            if st.button("🔀 Random", use_container_width=True):
                random_vals = np.random.random(n_outcomes)
                st.session_state.probs = (random_vals / random_vals.sum()).tolist()
                st.rerun()
            if st.button("🔄 Reset", use_container_width=True):
                if 'probs' in st.session_state:
                    del st.session_state.probs
                st.rerun()
    
    # Initialize session state
    if 'probs' not in st.session_state:
        st.session_state.probs = [1.0 / n_outcomes] * n_outcomes

    # Adjust size if n_outcomes changed
    if len(st.session_state.probs) != n_outcomes:
        st.session_state.probs = [1.0 / n_outcomes] * n_outcomes

    # Input Section with Professional Layout
    st.markdown('<h2 class="subsection-header">📥 Probability Distribution</h2>', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">
                Adjust the sliders below to create your probability distribution. Each value represents P(Xᵢ).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Create responsive slider grid
        cols = st.columns(min(n_outcomes, 4))
        probs = []

        for i in range(n_outcomes):
            col_idx = i % len(cols)
            with cols[col_idx]:
                with st.container():
                    prob = st.slider(
                        f"P(X{i+1})",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.probs[i]),
                        step=0.01,
                        key=f"entropy_p{i}",
                        help=f"Probability for outcome X{i+1}"
                    )
                    probs.append(prob)

                    # Show current value
                    st.caption(f"Current: {prob:.3f}")

    # Normalize if requested
    if auto_normalize:
        probs = normalize_distribution(probs)
        st.session_state.probs = probs

    # Validation and Results
    prob_sum = sum(probs)

    # Results Section
    st.markdown('<h2 class="subsection-header">📊 Analysis Results</h2>', unsafe_allow_html=True)

    # Validation Warning
    if abs(prob_sum - 1.0) > 0.01:
        st.markdown("""
        <div class="warning-card">
            <strong>⚠️ Validation Warning</strong><br>
            Probabilities sum to {:.3f}, not 1.0. Enable auto-normalization or adjust values manually.
        </div>
        """.format(prob_sum), unsafe_allow_html=True)

    # Calculate metrics
    h = calculate_entropy(probs, base=2)
    max_h = np.log2(n_outcomes)
    entropy_ratio = h / max_h if max_h > 0 else 0
    interpretation, color = get_entropy_interpretation(h, max_h)

    # Professional Metrics Display
    metrics_cols = st.columns(4)
    metric_data = [
        ("🎯 Entropy (H)", f"{h:.4f}", "bits", "Primary entropy measure"),
        ("📈 Max Entropy", f"{max_h:.4f}", "bits", "Theoretical maximum"),
        ("📊 Ratio", f"{entropy_ratio:.1%}", "", "Entropy efficiency"),
        ("✅ Sum", f"{prob_sum:.4f}", "", "Probability validation")
    ]

    for i, (label, value, unit, help_text) in enumerate(metric_data):
        with metrics_cols[i]:
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value} <span style="font-size: 0.8rem; font-weight: 400;">{unit}</span></div>
                    <div style="font-size: 0.8rem; color: #718096; margin-top: 0.25rem;">{help_text}</div>
                </div>
                """, unsafe_allow_html=True)

    # Interpretation Card
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {color}; margin-top: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">🎯 Interpretation</h4>
        <p style="margin: 0; font-size: 1.1rem; font-weight: 500;">{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Visualizations
    st.markdown('<h2 class="subsection-header">📈 Visual Analysis</h2>', unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        with st.container():
            st.markdown("""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #2d3748;">📊 Probability Distribution</h4>
                <p style="margin: 0.5rem 0; color: #718096; font-size: 0.9rem;">Visual representation of your probability values</p>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced probability distribution chart
            fig1 = go.Figure()

            # Create gradient colors
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
            bar_colors = colors[:n_outcomes]

            fig1.add_trace(go.Bar(
                x=[f'X{i+1}' for i in range(n_outcomes)],
                y=probs,
                marker=dict(
                    color=bar_colors,
                    line=dict(color='rgba(255,255,255,0.8)', width=2),
                    opacity=0.8
                ),
                text=[f'{p:.3f}' for p in probs],
                textposition='outside',
                textfont=dict(size=12, color='#2d3748'),
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.4f}<extra></extra>',
                name='Probability'
            ))

            # Add sum line
            fig1.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#e53e3e",
                annotation_text="Sum = 1.0",
                annotation_position="top right",
                opacity=0.7
            )

            fig1.update_layout(
                title=dict(
                    text="Probability Distribution",
                    font=dict(size=16, color='#2d3748', weight='bold'),
                    x=0.5
                ),
                xaxis=dict(
                    title="Outcomes",
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="Probability",
                    tickfont=dict(size=12),
                    range=[0, 1.1],
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(248,250,252,0.8)',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig1, width='stretch')

    with viz_col2:
        with st.container():
            st.markdown("""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #2d3748;">🎯 Entropy Comparison</h4>
                <p style="margin: 0.5rem 0; color: #718096; font-size: 0.9rem;">Current entropy vs theoretical maximum</p>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced entropy comparison chart
            fig2 = go.Figure()

            entropy_data = ['Current', 'Maximum']
            entropy_values = [h, max_h]
            entropy_colors = ['#667eea', '#e2e8f0']

            fig2.add_trace(go.Bar(
                x=entropy_data,
                y=entropy_values,
                marker=dict(
                    color=entropy_colors,
                    line=dict(color='rgba(255,255,255,0.8)', width=2),
                    opacity=0.9
                ),
                text=[f'{v:.4f}' for v in entropy_values],
                textposition='outside',
                textfont=dict(size=12, color='#2d3748', weight='bold'),
                hovertemplate='<b>%{x} Entropy</b><br>Value: %{y:.4f} bits<extra></extra>',
                name='Entropy'
            ))

            # Add ratio annotation
            fig2.add_annotation(
                x=0.5,
                y=max(entropy_values) * 0.9,
                text=f"Efficiency: {entropy_ratio:.1%}",
                showarrow=False,
                font=dict(size=12, color='#4a5568'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e2e8f0',
                borderwidth=1,
                borderpad=4
            )

            fig2.update_layout(
                title=dict(
                    text="Entropy Analysis",
                    font=dict(size=16, color='#2d3748', weight='bold'),
                    x=0.5
                ),
                xaxis=dict(
                    title="",
                    tickfont=dict(size=12, color='#4a5568')
                ),
                yaxis=dict(
                    title="Entropy (bits)",
                    tickfont=dict(size=12),
                    range=[0, max(max_h * 1.3, 3)],
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(248,250,252,0.8)',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig2, width='stretch')
    
    # Professional Properties Section
    st.markdown('<h2 class="subsection-header">📚 Information Theory Insights</h2>', unsafe_allow_html=True)

    # Create professional property cards
    prop_col1, prop_col2, prop_col3 = st.columns(3)

    with prop_col1:
        with st.container():
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0; color: #38a169;">📈 Maximum Entropy</h4>
                <ul style="margin: 0; padding-left: 1.2rem; color: #4a5568;">
                    <li>Uniform distribution (equal probabilities)</li>
                    <li>All outcomes equally likely</li>
                    <li><strong>Highest uncertainty</strong></li>
                    <li>Maximum information content</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with prop_col2:
        with st.container():
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0; color: #e53e3e;">📉 Minimum Entropy</h4>
                <ul style="margin: 0; padding-left: 1.2rem; color: #4a5568;">
                    <li>Delta distribution (one outcome = 1)</li>
                    <li>Certain outcome (p=1)</li>
                    <li><strong>Zero uncertainty</strong></li>
                    <li>Perfect predictability</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with prop_col3:
        with st.container():
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0; color: #3182ce;">🎯 Your Distribution</h4>
                <ul style="margin: 0; padding-left: 1.2rem; color: #4a5568;">
                    <li>Entropy: <strong>{h:.3f} bits</strong></li>
                    <li>Efficiency: <strong>{entropy_ratio:.1%}</strong></li>
                    <li>Information content: <strong>{'High' if entropy_ratio > 0.7 else 'Medium' if entropy_ratio > 0.4 else 'Low'}</strong></li>
                    <li>Predictability: <strong>{'Low' if entropy_ratio > 0.7 else 'Medium' if entropy_ratio > 0.4 else 'High'}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# MODE 2: KL DIVERGENCE CALCULATOR
elif mode == "KL Divergence Calculator":
    st.markdown('<h1 class="section-header">📏 KL Divergence Analysis</h1>', unsafe_allow_html=True)

    # Mathematical Foundation
    st.markdown("""
    <div class="formula-box">
        <h3 style="margin-top: 0; color: #2d3748;">Kullback-Leibler Divergence Formula</h3>
        <div style="font-size: 1.5rem; margin: 1rem 0;">
            $$D_{KL}(P \\parallel Q) = \\sum_{i=1}^{n} P(x_i) \\log \\frac{P(x_i)}{Q(x_i)}$$
        </div>
        <p style="margin: 0; color: #718096;"><em>Measures how distribution P differs from reference distribution Q</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Important Note about Asymmetry
    st.markdown("""
    <div class="warning-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #d69e2e;">⚠️ Important Property</h4>
        <p style="margin: 0;"><strong>KL Divergence is NOT symmetric:</strong> D(P||Q) ≠ D(Q||P)<br>
        The "distance" from P to Q is different from Q to P.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.subheader("Distribution Settings")
    n_outcomes_kl = st.sidebar.slider("Number of Outcomes", min_value=2, max_value=6, value=2, step=1)
    auto_normalize_kl = st.sidebar.checkbox("Auto-normalize probabilities", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Examples")
    if st.sidebar.button("🎯 Identical Distributions"):
        uniform = [1.0 / n_outcomes_kl] * n_outcomes_kl
        st.session_state.probs_p = uniform
        st.session_state.probs_q = uniform
    if st.sidebar.button("🎲 Random Distributions"):
        p_vals = np.random.random(n_outcomes_kl)
        q_vals = np.random.random(n_outcomes_kl)
        st.session_state.probs_p = (p_vals / p_vals.sum()).tolist()
        st.session_state.probs_q = (q_vals / q_vals.sum()).tolist()
    if st.sidebar.button("⚖️ Uniform vs Peaked"):
        st.session_state.probs_p = [1.0 / n_outcomes_kl] * n_outcomes_kl
        probs_q = [0.1] * n_outcomes_kl
        probs_q[0] = 1.0 - (n_outcomes_kl - 1) * 0.1
        st.session_state.probs_q = probs_q
    if st.sidebar.button("🔄 Reset"):
        if 'probs_p' in st.session_state:
            del st.session_state.probs_p
        if 'probs_q' in st.session_state:
            del st.session_state.probs_q
    
    # Initialize session state
    if 'probs_p' not in st.session_state:
        st.session_state.probs_p = [1.0 / n_outcomes_kl] * n_outcomes_kl
    if 'probs_q' not in st.session_state:
        st.session_state.probs_q = [1.0 / n_outcomes_kl] * n_outcomes_kl
    
    # Adjust size if n_outcomes changed
    if len(st.session_state.probs_p) != n_outcomes_kl:
        st.session_state.probs_p = [1.0 / n_outcomes_kl] * n_outcomes_kl
    if len(st.session_state.probs_q) != n_outcomes_kl:
        st.session_state.probs_q = [1.0 / n_outcomes_kl] * n_outcomes_kl
    
    # Create two columns for P and Q
    col_p, col_q = st.columns(2)
    
    with col_p:
        st.subheader("📥 Distribution P (True)")
        probs_p = []
        for i in range(n_outcomes_kl):
            prob = st.slider(
                f"P(X{i+1})",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.probs_p[i]),
                step=0.01,
                key=f"kl_p{i}"
            )
            probs_p.append(prob)
    
    with col_q:
        st.subheader("📥 Distribution Q (Reference)")
        probs_q = []
        for i in range(n_outcomes_kl):
            prob = st.slider(
                f"Q(X{i+1})",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.probs_q[i]),
                step=0.01,
                key=f"kl_q{i}"
            )
            probs_q.append(prob)
    
    # Normalize if requested
    if auto_normalize_kl:
        probs_p = normalize_distribution(probs_p)
        probs_q = normalize_distribution(probs_q)
        st.session_state.probs_p = probs_p
        st.session_state.probs_q = probs_q
    
    # Validation
    sum_p = sum(probs_p)
    sum_q = sum(probs_q)
    
    if abs(sum_p - 1.0) > 0.01 or abs(sum_q - 1.0) > 0.01:
        st.warning(f"⚠️ Warning: P sums to {sum_p:.3f}, Q sums to {sum_q:.3f}. Enable auto-normalize or adjust values.")
    
    # Calculate KL divergences
    kl_pq = calculate_kl_divergence(probs_p, probs_q)  # D(P||Q)
    kl_qp = calculate_kl_divergence(probs_q, probs_p)  # D(Q||P)
    asymmetry = abs(kl_pq - kl_qp)
    
    interpretation_pq, color_pq = get_kl_interpretation(kl_pq)
    interpretation_qp, color_qp = get_kl_interpretation(kl_qp)
    
    # Display metrics
    st.markdown("---")
    st.subheader("📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("D(P||Q)", f"{kl_pq:.6f}")
    with col2:
        st.metric("D(Q||P)", f"{kl_qp:.6f}")
    with col3:
        st.metric("Asymmetry", f"{asymmetry:.6f}")
    with col4:
        st.metric("Symmetric?", "No" if asymmetry > 0.001 else "Yes")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: {color_pq};">
            <b>D(P||Q) Interpretation:</b> {interpretation_pq}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: {color_qp};">
            <b>D(Q||P) Interpretation:</b> {interpretation_qp}
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    
    # Create 3-column layout for visualizations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Plot 1: Distribution comparison
        fig1 = go.Figure()
        
        x_labels = [f'X{i+1}' for i in range(n_outcomes_kl)]
        
        fig1.add_trace(go.Bar(
            x=x_labels,
            y=probs_p,
            name='P (True)',
            marker=dict(color='#FF6B6B', line=dict(color='black', width=2)),
            text=[f'{p:.3f}' for p in probs_p],
            textposition='outside'
        ))
        
        fig1.add_trace(go.Bar(
            x=x_labels,
            y=probs_q,
            name='Q (Reference)',
            marker=dict(color='#4ECDC4', line=dict(color='black', width=2)),
            text=[f'{q:.3f}' for q in probs_q],
            textposition='outside'
        ))
        
        fig1.update_layout(
            title=dict(text="Distribution Comparison", font=dict(size=16, color='#1f77b4')),
            xaxis_title="Outcome",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1.1]),
            barmode='group',
            height=400,
            template="plotly_white",
            legend=dict(x=0.7, y=1.0)
        )
        
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        # Plot 2: KL divergences (both directions)
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=['D(P||Q)', 'D(Q||P)'],
            y=[kl_pq, kl_qp],
            marker=dict(
                color=['#FF6B6B', '#4ECDC4'],
                line=dict(color='black', width=2)
            ),
            text=[f'{kl_pq:.4f}', f'{kl_qp:.4f}'],
            textposition='outside'
        ))
        
        fig2.update_layout(
            title=dict(text="KL Divergence (Asymmetry)", font=dict(size=16, color='#1f77b4')),
            xaxis_title="",
            yaxis_title="KL Divergence",
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, width='stretch')
    
    with col3:
        # Plot 3: Absolute difference
        abs_diff = [abs(p - q) for p, q in zip(probs_p, probs_q)]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            x=x_labels,
            y=abs_diff,
            marker=dict(
                color='#95E1D3',
                line=dict(color='black', width=2)
            ),
            text=[f'{d:.3f}' for d in abs_diff],
            textposition='outside'
        ))
        
        fig3.update_layout(
            title=dict(text="Absolute Difference |P-Q|", font=dict(size=16, color='#1f77b4')),
            xaxis_title="Outcome",
            yaxis_title="|P - Q|",
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig3, width='stretch')
    
    # Warning about asymmetry
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <b>⚠️ Important Property: Asymmetry</b><br>
        KL Divergence is <b>NOT symmetric</b>: D(P||Q) ≠ D(Q||P)<br>
        This means KL divergence is not a true distance metric. The "distance" from P to Q 
        is different from the "distance" from Q to P.
    </div>
    """, unsafe_allow_html=True)

# MODE 3: COMPARISON EXAMPLES
elif mode == "Comparison Examples":
    st.markdown('<p class="sub-header">📚 Comparison Examples</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore predefined examples to understand how entropy and KL divergence behave 
    with different probability distributions.
    """)
    
    # Define example distributions
    examples = {
        "Uniform (Max Entropy)": [0.333, 0.333, 0.334],
        "Slightly Skewed": [0.2, 0.5, 0.3],
        "Concentrated": [0.1, 0.8, 0.1],
        "Certain (Min Entropy)": [0.0, 1.0, 0.0]
    }
    
    # Calculate entropy for all examples
    st.subheader("🎲 Entropy Comparison")
    
    entropy_data = []
    for name, dist in examples.items():
        h = calculate_entropy(dist, base=2)
        max_h = np.log2(len(dist))
        ratio = h / max_h if max_h > 0 else 0
        entropy_data.append({
            "Distribution": name,
            "Entropy (bits)": f"{h:.6f}",
            "Max Entropy": f"{max_h:.6f}",
            "Ratio": f"{ratio:.1%}"
        })
    
    df_entropy = pd.DataFrame(entropy_data)
    st.dataframe(df_entropy, width='stretch')
    
    # Visualization of entropies
    fig_entropy = go.Figure()
    
    names = list(examples.keys())
    entropies = [calculate_entropy(examples[name], base=2) for name in names]
    max_entropy_val = np.log2(3)
    
    fig_entropy.add_trace(go.Bar(
        x=names,
        y=entropies,
        marker=dict(
            color=entropies,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Entropy"),
            line=dict(color='black', width=2)
        ),
        text=[f'{e:.3f}' for e in entropies],
        textposition='outside'
    ))
    
    # Add max entropy line
    fig_entropy.add_hline(
        y=max_entropy_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Entropy = {max_entropy_val:.3f}",
        annotation_position="right"
    )
    
    fig_entropy.update_layout(
        title="Entropy Comparison Across Distributions",
        xaxis_title="Distribution Type",
        yaxis_title="Entropy (bits)",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_entropy, width='stretch')
    
    # KL Divergence comparisons
    st.markdown("---")
    st.subheader("📏 KL Divergence Comparison")
    
    st.markdown("""
    Compare different distributions against a uniform reference distribution.
    """)
    
    # Reference distribution (uniform)
    reference = [0.333, 0.333, 0.334]
    
    kl_data = []
    for name, dist in examples.items():
        kl_forward = calculate_kl_divergence(dist, reference)
        kl_backward = calculate_kl_divergence(reference, dist)
        asymmetry = abs(kl_forward - kl_backward)
        
        kl_data.append({
            "Distribution": name,
            "D(P||Uniform)": f"{kl_forward:.6f}",
            "D(Uniform||P)": f"{kl_backward:.6f}",
            "Asymmetry": f"{asymmetry:.6f}"
        })
    
    df_kl = pd.DataFrame(kl_data)
    st.dataframe(df_kl, width='stretch')
    
    # Visualization of KL divergences
    fig_kl = make_subplots(
        rows=1, cols=2,
        subplot_titles=("D(P||Uniform)", "D(Uniform||P)")
    )
    
    kl_forward_vals = [calculate_kl_divergence(examples[name], reference) for name in names]
    kl_backward_vals = [calculate_kl_divergence(reference, examples[name]) for name in names]
    
    fig_kl.add_trace(
        go.Bar(
            x=names,
            y=kl_forward_vals,
            marker=dict(color='#FF6B6B', line=dict(color='black', width=2)),
            text=[f'{v:.4f}' for v in kl_forward_vals],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig_kl.add_trace(
        go.Bar(
            x=names,
            y=kl_backward_vals,
            marker=dict(color='#4ECDC4', line=dict(color='black', width=2)),
            text=[f'{v:.4f}' for v in kl_backward_vals],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_kl.update_xaxes(title_text="Distribution", row=1, col=1)
    fig_kl.update_xaxes(title_text="Distribution", row=1, col=2)
    fig_kl.update_yaxes(title_text="KL Divergence", row=1, col=1)
    fig_kl.update_yaxes(title_text="KL Divergence", row=1, col=2)
    
    fig_kl.update_layout(
        title_text="KL Divergence: Demonstrating Asymmetry",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_kl, width='stretch')
    
    # Distribution visualizations
    st.markdown("---")
    st.subheader("📊 Distribution Visualizations")
    
    fig_dists = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(examples.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (name, dist) in enumerate(examples.items()):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig_dists.add_trace(
            go.Bar(
                x=[f'X{i+1}' for i in range(len(dist))],
                y=dist,
                marker=dict(color=colors_list, line=dict(color='black', width=1.5)),
                text=[f'{p:.2f}' for p in dist],
                textposition='outside',
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig_dists.update_yaxes(range=[0, 1.1], row=row, col=col)
    
    fig_dists.update_layout(
        title_text="All Example Distributions",
        height=700,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_dists, width='stretch')

# Professional Footer
st.markdown("""
<div class="footer">
    <div style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1.5rem;">🔬 Information Theory Laboratory</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1rem; color: #4a5568;">
            An interactive platform for exploring fundamental concepts in information science and probability theory.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">📊</div>
                <div style="font-weight: 600; color: #2d3748;">Entropy Analysis</div>
                <div class="footer-text">Measure uncertainty in distributions</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🔄</div>
                <div style="font-weight: 600; color: #2d3748;">KL Divergence</div>
                <div class="footer-text">Compare probability distributions</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🎓</div>
                <div style="font-weight: 600; color: #2d3748;">Educational</div>
                <div class="footer-text">Learn through interactive exploration</div>
            </div>
        </div>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem;">
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Built with:</strong> Streamlit • Plotly • NumPy • SciPy
            </p>
            <p class="footer-text" style="margin: 0.5rem 0;">
                <strong>Educational Purpose:</strong> Interactive demonstration of information theory concepts
            </p>
            <p class="footer-text" style="margin: 0.5rem 0; font-style: italic;">
                💡 <strong>Tip:</strong> Experiment with different probability distributions to build your intuition about information theory!
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
