import streamlit as st
from transformers import pipeline
import torch

# Set page config
st.set_page_config(
    page_title="Chapter 20: Advanced Question Answering",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("Chapter 20: Advanced Question Answering with Transformers")
st.markdown("""
This application demonstrates advanced NLP concepts using a pre-trained Question Answering model.
The model can extract answers from a given context based on your questions.
""")

# Sidebar for configuration
st.sidebar.header("Model Configuration")
st.sidebar.markdown("""
**Model:** distilbert-base-cased-distilled-squad

This is a distilled version of BERT fine-tuned on the SQuAD dataset for question answering tasks.
""")

# Initialize the QA pipeline with caching
@st.cache_resource
def load_qa_model():
    """Load the Question Answering pipeline"""
    with st.spinner("Loading Question Answering model..."):
        qa_pipeline = pipeline(
            task="question-answering",
            model="distilbert-base-cased-distilled-squad"
        )
    return qa_pipeline

# Load model
try:
    qa_pipeline = load_qa_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {str(e)}")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Context")
    
    # Default context
    default_context = (
        "Alice met Bob at the park on Sunday. "
        "They played chess and had ice cream."
    )
    
    # Text area for context input
    context = st.text_area(
        "Enter the context (passage of text):",
        value=default_context,
        height=200,
        help="Provide the text passage from which the answer will be extracted"
    )

with col2:
    st.subheader("Ask a Question")
    
    # Default question
    default_question = "What did Alice and Bob do at the park?"
    
    # Text input for question
    question = st.text_input(
        "Enter your question:",
        value=default_question,
        help="Ask a question about the context"
    )

# Confidence threshold slider
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Minimum confidence score to display the answer"
)

# Button to generate answer
if st.button("Get Answer", type="primary"):
    if context.strip() == "":
        st.warning("Please provide a context.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Generating answer..."):
                # Generate answer
                result = qa_pipeline(
                    question=question,
                    context=context
                )
            
            # Display results
            st.divider()
            st.subheader("Results")
            
            # Check confidence threshold
            if result["score"] >= confidence_threshold:
                # Display answer in a nice box
                st.success("**Answer Found!**")
                
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown("### Answer:")
                    st.info(f"**{result['answer']}**")
                
                with col_b:
                    st.markdown("### Confidence:")
                    st.metric("Score", f"{result['score']:.4f}")
                    
                    # Visual confidence indicator
                    if result["score"] > 0.8:
                        st.success("High Confidence")
                    elif result["score"] > 0.5:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")
                
                # Additional details in expander
                with st.expander("View Additional Details"):
                    st.json(result)
            else:
                st.warning(f"Answer confidence ({result['score']:.4f}) is below the threshold ({confidence_threshold:.2f})")
                st.info(f"Answer: {result['answer']}")
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")

# Examples section
st.divider()
st.subheader("Try These Examples")

examples = [
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.",
        "question": "Who designed the Eiffel Tower?"
    },
    {
        "context": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant indentation.",
        "question": "When was Python first released?"
    },
    {
        "context": "The human brain contains approximately 86 billion neurons. These neurons communicate with each other through synapses, forming complex neural networks. The brain weighs about 3 pounds and uses about 20% of the body's total energy.",
        "question": "How many neurons are in the human brain?"
    }
]

for idx, example in enumerate(examples, 1):
    with st.expander(f"Example {idx}"):
        st.markdown(f"**Context:** {example['context']}")
        st.markdown(f"**Question:** {example['question']}")
        
        if st.button(f"Load Example {idx}", key=f"example_{idx}"):
            st.session_state['example_context'] = example['context']
            st.session_state['example_question'] = example['question']
            st.rerun()

# Apply example if loaded
if 'example_context' in st.session_state:
    st.info("Example loaded! Scroll up to see the context and question, then click 'Get Answer'.")

# Footer
st.divider()
st.markdown("""
### About Question Answering

Question Answering (QA) is an NLP task where the model extracts or generates answers to questions based on a given context. 
The model used here (DistilBERT) is trained on the SQuAD dataset and can handle extractive QA tasks where the answer 
is a span of text from the provided context.

**Key Features:**
- Extracts precise answers from context
- Provides confidence scores
- Works with various types of questions (Who, What, When, Where, Why, How)
""")
