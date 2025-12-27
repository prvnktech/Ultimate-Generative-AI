import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Page configuration
st.set_page_config(
    page_title="T5 Text Summarizer",
    page_icon="📝",
    layout="centered"
)

# Title and description
st.title("📝 T5 Text Summarizer")
st.markdown("Enter text below and get an AI-generated summary using the T5-small model.")

# Initialize model and tokenizer with caching
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    return tokenizer, model, device

# Load model
with st.spinner("Loading T5 model... This may take a moment on first run."):
    tokenizer, model, device = load_model()

st.success(f"✅ Model loaded successfully! Using: {device.upper()}")

# Sidebar for parameters
st.sidebar.header("⚙️ Generation Parameters")
max_length = st.sidebar.slider(
    "Max Summary Length",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="Maximum length of the generated summary"
)

num_beams = st.sidebar.slider(
    "Number of Beams",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    help="Higher values = better quality but slower generation"
)

# Text input
text_input = st.text_area(
    "Enter text to summarize:",
    height=200,
    placeholder="Type or paste your text here...",
    help="The model works best with text that is at least a few sentences long."
)

# Example texts
st.sidebar.header("📄 Example Texts")
if st.sidebar.button("Example 1: Technology"):
    text_input = "Artificial intelligence has made remarkable progress in recent years. Machine learning models can now perform tasks that were once thought to require human intelligence, such as image recognition, natural language processing, and game playing. Deep learning, a subset of machine learning, has been particularly successful. However, these systems still have limitations and challenges, including the need for large amounts of data, computational resources, and concerns about bias and interpretability."
    st.rerun()

if st.sidebar.button("Example 2: Nature"):
    text_input = "The Amazon rainforest is the largest tropical rainforest in the world, covering approximately 5.5 million square kilometers. It is home to an incredible diversity of plant and animal species, many of which are found nowhere else on Earth. The rainforest plays a crucial role in regulating global climate by absorbing carbon dioxide and producing oxygen. However, deforestation and climate change pose serious threats to this vital ecosystem."
    st.rerun()

# Generate summary button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    generate_button = st.button("🚀 Generate Summary", use_container_width=True)

# Generate summary
if generate_button:
    if text_input.strip():
        with st.spinner("Generating summary..."):
            try:
                # Prepare input with task prefix
                input_text = "summarize: " + text_input
                
                # Tokenize
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Generate summary
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                
                # Decode output
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display results
                st.success("✨ Summary generated successfully!")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Original Length", f"{len(text_input.split())} words")
                with col_b:
                    st.metric("Summary Length", f"{len(summary.split())} words")
                
                st.subheader("📋 Summary:")
                st.info(summary)
                
                # Show original text in expander
                with st.expander("📖 View Original Text"):
                    st.write(text_input)
                    
            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text to summarize.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Powered by T5-small from Hugging Face 🤗
    </div>
    """,
    unsafe_allow_html=True
)