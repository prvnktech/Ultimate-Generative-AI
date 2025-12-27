# TensorFlow GPT Lab

An interactive Streamlit application demonstrating TensorFlow implementations for GPT-2 model fine-tuning and inference.

## Features

- **TensorFlow Integration**: Explore GPT-2 implementation in TensorFlow/Keras
- **Forward Pass Visualization**: See how TF processes input through the model
- **Loss Calculation**: Learn how training loss is computed in TensorFlow
- **Interactive Experimentation**: Input your own text and observe model behavior
- **Cross-Framework Comparison**: Compare with PyTorch implementation

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application using Streamlit:

```
streamlit run Requirement-17.4.py
```

The web application will open in your browser, allowing you to:
- Enter custom text samples
- Run a forward pass through the TFGPT2LMHeadModel
- View tensor shapes and representations
- Observe calculated loss values and their interpretations

## Concepts Covered

- TensorFlow tensor operations
- TFGPT2LMHeadModel implementation
- Eager execution in TensorFlow 2.x
- Loss calculation for language models
- Cross-framework considerations (TF vs PyTorch)

## Technologies Used

- **Streamlit**: For the interactive web interface
- **TensorFlow**: Deep learning framework for model implementation
- **tf-keras**: For compatibility between Keras 3 and Transformers
- **Hugging Face Transformers**: For the TFGPT2LMHeadModel implementation
- **GPT-2**: OpenAI's language model architecture