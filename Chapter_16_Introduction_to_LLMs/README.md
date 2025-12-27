# Chapter 16: Introduction to Large Language Models (LLMs)

This chapter introduces Large Language Models (LLMs), which have revolutionized AI by demonstrating emergent capabilities in understanding and generating human-like text at scale.

## Overview

Large Language Models represent a paradigm shift in NLP, scaling transformer architectures to billions of parameters and unlocking remarkable abilities. This chapter covers:

- **What are LLMs?** - Understanding scale and emergence
- **Transformer Architecture** - Foundation of modern LLMs
- **Hugging Face Ecosystem** - Using pre-trained models
- **Model Inference** - Running LLMs efficiently
- **Practical Applications** - Real-world use cases

## Features

- **Hands-on Notebooks**: Practical demonstrations with popular LLMs
- **Multiple Models**: GPT, BERT, T5, and more
- **Hugging Face Integration**: Industry-standard library
- **Inference Techniques**: Optimize for speed and quality
- **API Usage**: Working with model APIs

## Files

- `chapter_16_huggingface_transformers.py` - Hugging Face transformers usage
- `chapter_16_ai_test.py` - LLM testing and evaluation
- `chapter_16_requirements.py` - Setup and requirements
- `Notebook/chapter_16_huggingface_transformers.ipynb` - Detailed transformers notebook
- `Notebook/chapter_16_ai_test.ipynb` - Testing notebook
- `Notebook/chapter_16_requirements.ipynb` - Requirements notebook

## Installation

```bash
pip install transformers torch accelerate sentencepiece protobuf
```

For advanced features:
```bash
pip install bitsandbytes optimum  # Quantization
pip install flash-attn  # Faster attention
```

## Usage

### Hugging Face Transformers

```bash
python Chapter_16_Introduction_to_LLMs/chapter_16_huggingface_transformers.py
```

Features:
- Load pre-trained models
- Text generation
- Classification tasks
- Model fine-tuning basics

### Testing LLMs

```bash
python Chapter_16_Introduction_to_LLMs/chapter_16_ai_test.py
```

Features:
- Evaluate model performance
- Compare different models
- Benchmark inference speed
- Quality assessment

### Jupyter Notebooks

```bash
jupyter notebook Chapter_16_Introduction_to_LLMs/Notebook/
```

## Key Concepts

### What Makes LLMs "Large"?

**Scale Comparison:**
- **Small Models**: 100M - 1B parameters (BERT-base, DistilGPT-2)
- **Medium Models**: 1B - 10B parameters (GPT-2, GPT-Neo)
- **Large Models**: 10B - 100B parameters (GPT-3, LLaMA-70B)
- **Massive Models**: 100B+ parameters (GPT-4, PaLM, Gemini)

### Emergent Capabilities

As models scale, new abilities emerge:
- **Few-Shot Learning**: Learn from just a few examples
- **In-Context Learning**: Adapt without parameter updates
- **Chain-of-Thought**: Step-by-step reasoning
- **Instruction Following**: Understand and execute commands
- **Multi-Task Learning**: Perform diverse tasks

### Transformer Architecture Review

```
Input Tokens
↓
Token Embeddings + Positional Encodings
↓
[Multi-Head Self-Attention
↓
Add & Norm
↓
Feed-Forward Network
↓
Add & Norm] × N layers
↓
Output Layer
↓
Generated Tokens
```

**Key Components:**
- **Self-Attention**: Model relationships between all tokens
- **Feed-Forward**: Process each position independently
- **Layer Normalization**: Stabilize training
- **Residual Connections**: Enable deep networks

## What You'll Learn

1. **LLM Fundamentals**
   - Architecture variations (GPT, BERT, T5)
   - Training objectives (CLM, MLM, Seq2Seq)
   - Tokenization methods
   - Model sizes and trade-offs

2. **Using Pre-trained Models**
   - Hugging Face Hub
   - Loading and configuring models
   - Inference pipelines
   - Memory management

3. **Practical Applications**
   - Text generation
   - Question answering
   - Summarization
   - Translation
   - Classification

4. **Optimization Techniques**
   - Quantization (4-bit, 8-bit)
   - Model pruning
   - Knowledge distillation
   - Efficient inference

## Popular LLM Families

### GPT Family (Decoder-Only)
**Architecture**: Auto-regressive generation
**Use Case**: Text generation, completion

- **GPT-2**: 117M - 1.5B params, open-source
- **GPT-3**: 175B params, API access
- **GPT-3.5**: ChatGPT base
- **GPT-4**: Multimodal, reasoning

### BERT Family (Encoder-Only)
**Architecture**: Masked language modeling
**Use Case**: Classification, NER, QA

- **BERT-base**: 110M params
- **BERT-large**: 340M params
- **RoBERTa**: Optimized BERT
- **ALBERT**: Parameter sharing

### T5 Family (Encoder-Decoder)
**Architecture**: Text-to-text framework
**Use Case**: All NLP tasks as text generation

- **T5-small**: 60M params
- **T5-base**: 220M params
- **T5-large**: 770M params
- **Flan-T5**: Instruction-tuned

### Open-Source LLMs

- **LLaMA (Meta)**: 7B - 70B params
- **Falcon**: 7B - 180B params
- **Mistral**: 7B params, high quality
- **Gemma (Google)**: 2B - 7B params

## Hugging Face Transformers

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0])
print(generated_text)
```

### Using Pipelines (High-Level API)

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# Question answering
qa = pipeline("question-answering")
result = qa(question="What is AI?", context="AI is artificial intelligence...")

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer("Long article text...", max_length=100)

# Translation
translator = pipeline("translation_en_to_fr", model="t5-base")
result = translator("Hello, how are you?")
```

## Memory Optimization

### 4-bit Quantization

```python
from transformers import BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load quantized model (75% memory reduction)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 8-bit Quantization

```python
# Simpler, 50% memory reduction
model = AutoModelForCausalLM.from_pretrained(
    "gpt2-xl",
    load_in_8bit=True,
    device_map="auto"
)
```

## Generation Parameters

### Temperature
Control randomness:
```python
# temperature = 0.0  → Deterministic (greedy)
# temperature = 0.7  → Balanced (recommended)
# temperature = 1.0  → Standard sampling
# temperature = 1.5+ → Very creative/random
outputs = model.generate(
    inputs,
    temperature=0.7,
    do_sample=True
)
```

### Top-k and Top-p

```python
# Top-k: Sample from top k tokens
outputs = model.generate(
    inputs,
    do_sample=True,
    top_k=50
)

# Top-p (nucleus): Sample from smallest set with cumulative prob ≥ p
outputs = model.generate(
    inputs,
    do_sample=True,
    top_p=0.9
)

# Combine both
outputs = model.generate(
    inputs,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
```

### Repetition Penalty

```python
# Prevent repetitive text (1.0 = no penalty, 1.5 = strong penalty)
outputs = model.generate(
    inputs,
    repetition_penalty=1.2
)
```

## Training vs Fine-Tuning vs Prompting

| Approach | Data Needed | Compute | Use Case |
|----------|-------------|---------|----------|
| **Pre-training** | Billions of tokens | Thousands of GPUs | Create base model |
| **Fine-tuning** | 1K-1M examples | Few GPUs, days | Domain adaptation |
| **Prompt Engineering** | 0-10 examples | None (inference only) | Quick adaptation |
| **PEFT (LoRA)** | 100-10K examples | Single GPU, hours | Efficient fine-tuning |

## Common Use Cases

### 1. Text Generation
```python
# Story writing, content creation
generator("Write a story about", max_length=200)
```

### 2. Question Answering
```python
# Extract answers from context
qa(question="Who invented the telephone?", 
   context="Alexander Graham Bell invented the telephone in 1876...")
```

### 3. Summarization
```python
# Condense long documents
summarizer(article_text, min_length=30, max_length=100)
```

### 4. Translation
```python
# Language translation
translator("Hello, world!")
```

### 5. Code Generation
```python
# Generate code from descriptions
code_gen("Function to calculate fibonacci numbers in Python")
```

## Evaluation Metrics

### Perplexity
- Measures model confidence
- Lower is better
- `perplexity = exp(loss)`

### Accuracy
- Task-specific (classification, QA)
- Percentage of correct predictions

### Human Evaluation
- Fluency, coherence, relevance
- User preference studies

## Best Practices

### Prompt Engineering Tips

1. **Be Specific**: Clearly state what you want
2. **Provide Examples**: Few-shot learning works well
3. **Use Structure**: Format inputs consistently
4. **Add Context**: Background information helps
5. **Iterate**: Refine prompts based on outputs

### Inference Optimization

1. **Batching**: Process multiple inputs together
2. **Caching**: Save and reuse KV-cache
3. **Quantization**: Reduce precision (4-bit, 8-bit)
4. **Model Selection**: Choose size based on needs
5. **Hardware**: Use GPU/TPU when available

## Challenges and Limitations

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Hallucination** | Generate false information | Retrieval-augmented generation (RAG) |
| **Bias** | Reflect training data biases | Careful curation, alignment |
| **Cost** | Expensive inference | Smaller models, caching |
| **Privacy** | Data leakage concerns | On-premise deployment |
| **Interpretability** | Black-box nature | Attention visualization, probing |

## Advanced Topics

- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA, QLoRA, Adapters
- **Retrieval-Augmented Generation (RAG)**: Combine with knowledge bases
- **Multi-Modal LLMs**: Text + images (CLIP, LLaVA)
- **Agent Systems**: LLMs with tools and memory
- **Alignment**: RLHF, DPO, constitutional AI

## Resources

### Documentation
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [BERT](https://arxiv.org/abs/1810.04805) - Bidirectional Encoders
- [GPT-3](https://arxiv.org/abs/2005.14165) - Language Models are Few-Shot Learners
- [LLaMA](https://arxiv.org/abs/2302.13971) - Open Foundation Models

### Tutorials
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1)

## Next Steps

After mastering LLM basics:
- **Chapter 17**: GPT Architecture Deep Dive
- **Chapter 18**: LangChain Applications
- **Chapter 20**: Advanced Concepts (Question Answering, etc.)

---

**Explore LLMs!** Start with the notebooks and discover the power of large language models!
