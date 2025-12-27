# Chapter 12: Generative Models in NLP

This chapter explores how generative models are applied to Natural Language Processing (NLP), covering text generation, language modeling, and sequence-to-sequence tasks.

## Overview

Generative models have revolutionized NLP, enabling machines to generate coherent text, translate languages, and understand context. This chapter covers:

- **Language Models** - Predicting and generating text sequences
- **RNN-Based Generation** - Recurrent architectures for text
- **Transformer-Based Models** - Attention mechanisms for NLP
- **Text Generation Techniques** - Sampling strategies and decoding methods
- **Practical Applications** - Chatbots, translation, summarization

## Features

- **Two Interactive Exercises**:
  - Exercise 12.1: RNN/LSTM-based text generation
  - Exercise 12.2: Transformer-based language modeling
- **Streamlit Applications**: Interactive text generation interfaces
- **Multiple Sampling Methods**: Greedy, beam search, top-k, nucleus sampling
- **Pre-trained Models**: Fine-tuning and transfer learning

## Files

- `streamlit_chapter_12_exercise_12_1.py` - RNN text generation application
- `streamlit_chapter_12_exercise_12_2.py` - Transformer text generation application
- `Notebook/chapter_12_exercise_12_1.ipynb` - RNN detailed notebook
- `Notebook/chapter_12_exercise_12_2.ipynb` - Transformer detailed notebook
- `12.1_train.txt` - Training data for Exercise 12.1
- `12.1_val.txt` - Validation data for Exercise 12.1

## Installation

```bash
pip install torch transformers streamlit matplotlib numpy
```

For advanced features:
```bash
pip install tokenizers datasets sentencepiece
```

## Usage

### Exercise 12.1: RNN Text Generation

```bash
streamlit run Chapter_12_Generative_Models_in_NLP/streamlit_chapter_12_exercise_12_1.py
```

Features:
- Train character-level or word-level RNN
- Generate text with different temperatures
- Experiment with LSTM/GRU architectures
- Visualize training progress

### Exercise 12.2: Transformer Models

```bash
streamlit run Chapter_12_Generative_Models_in_NLP/streamlit_chapter_12_exercise_12_2.py
```

Features:
- Use pre-trained GPT models
- Fine-tune on custom datasets
- Multiple decoding strategies
- Control generation parameters

### Jupyter Notebooks

```bash
jupyter notebook Chapter_12_Generative_Models_in_NLP/Notebook/
```

## Key Concepts

### Language Modeling

**Goal**: Model probability distribution over text sequences

```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)
```

**Applications:**
- Text generation
- Next word prediction
- Speech recognition
- Machine translation

### RNN Architecture for Text

```
Input: Text sequence (word embeddings)
↓
Embedding Layer
↓
RNN/LSTM/GRU Layers (with hidden states)
↓
Dense Layer
↓
Softmax
↓
Output: Probability distribution over vocabulary
```

**Key Components:**
- **Embedding Layer**: Convert words to dense vectors
- **Recurrent Layers**: Capture sequential dependencies
- **Hidden States**: Maintain context information
- **Output Layer**: Predict next token

### Transformer Architecture

```
Input: Token sequence
↓
Positional Encoding + Token Embeddings
↓
Multi-Head Self-Attention (parallel processing)
↓
Feed-Forward Networks
↓
Layer Normalization
↓
Output: Contextualized representations
```

**Advantages over RNNs:**
- Parallel processing (faster training)
- Better long-range dependencies
- More expressive representations
- Scalability to large models

## What You'll Learn

1. **Text Generation Fundamentals**
   - Tokenization strategies
   - Vocabulary construction
   - Embedding techniques
   - Sequence modeling

2. **RNN-Based Models**
   - Vanilla RNN limitations
   - LSTM and GRU improvements
   - Bidirectional RNNs
   - Handling variable-length sequences

3. **Transformer Models**
   - Self-attention mechanism
   - Positional encodings
   - Multi-head attention
   - GPT architecture (decoder-only)

4. **Generation Strategies**
   - Greedy decoding
   - Beam search
   - Top-k sampling
   - Nucleus (top-p) sampling
   - Temperature scaling

## Text Generation Methods

### 1. Greedy Decoding
Always pick the most likely next token:
```python
def greedy_decode(model, start_tokens, max_length):
    for _ in range(max_length):
        logits = model(tokens)
        next_token = argmax(logits)
        tokens.append(next_token)
    return tokens
```

**Pros**: Fast, deterministic
**Cons**: Repetitive, can miss better sequences

### 2. Beam Search
Keep top-k hypotheses at each step:
```python
def beam_search(model, start_tokens, beam_width=5):
    beams = [(start_tokens, 0.0)]  # (sequence, score)
    for _ in range(max_length):
        candidates = []
        for seq, score in beams:
            logits = model(seq)
            top_k = topk(logits, beam_width)
            for token, prob in top_k:
                candidates.append((seq + [token], score + log(prob)))
        beams = sorted(candidates, key=lambda x: x[1])[:beam_width]
    return beams[0][0]
```

**Pros**: Better quality than greedy
**Cons**: Still can be repetitive, computational cost

### 3. Top-k Sampling
Sample from top k most likely tokens:
```python
def top_k_sampling(logits, k=40, temperature=1.0):
    # Scale by temperature
    logits = logits / temperature
    # Get top k
    top_k_logits, top_k_indices = topk(logits, k)
    # Sample from top k
    probs = softmax(top_k_logits)
    next_token = sample(top_k_indices, probs)
    return next_token
```

**Pros**: More diverse, creative
**Cons**: Can generate nonsense with high k

### 4. Nucleus (Top-p) Sampling
Sample from smallest set with cumulative probability ≥ p:
```python
def nucleus_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = sort(logits, descending=True)
    cumulative_probs = cumsum(softmax(sorted_logits))
    # Find cutoff
    cutoff_index = (cumulative_probs > p).argmax()
    # Sample from nucleus
    nucleus_logits = sorted_logits[:cutoff_index+1]
    nucleus_indices = sorted_indices[:cutoff_index+1]
    probs = softmax(nucleus_logits)
    next_token = sample(nucleus_indices, probs)
    return next_token
```

**Pros**: Adapts to distribution, balanced creativity
**Cons**: Hyperparameter tuning needed

### Temperature Scaling

Control randomness in sampling:
```
temperature = 0.0  → Greedy (deterministic)
temperature = 1.0  → Standard sampling
temperature > 1.0  → More random (creative)
temperature < 1.0  → More focused (conservative)
```

## Training Strategies

### Character-Level vs Word-Level

| Aspect | Character-Level | Word-Level |
|--------|-----------------|------------|
| **Vocabulary Size** | Small (~100) | Large (10K-50K) |
| **Sequence Length** | Very long | Moderate |
| **OOV Handling** | No OOV issues | Requires special tokens |
| **Training Speed** | Slower | Faster |
| **Use Case** | Morphology, names | Semantic tasks |

### Tokenization Methods

1. **Character**: Split into individual characters
2. **Word**: Split by whitespace/punctuation
3. **Subword (BPE)**: Balance between char and word
4. **WordPiece**: Used in BERT
5. **SentencePiece**: Language-agnostic

### Loss Function

**Cross-Entropy Loss:**
```python
loss = -∑ᵢ yᵢ log(ŷᵢ)
```

Where:
- yᵢ: True next token (one-hot)
- ŷᵢ: Predicted probability distribution

### Training Tips

- **Gradient Clipping**: Prevent exploding gradients (clip to 5.0)
- **Learning Rate Scheduling**: Warm-up then decay
- **Teacher Forcing**: Use ground truth during training
- **Regularization**: Dropout, weight decay
- **Batch Size**: 32-128 typical

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Repetition | Same phrases repeated | Use nucleus sampling, add repetition penalty |
| Incoherence | Nonsensical text | Reduce temperature, use beam search |
| Boring output | Generic, predictable | Increase temperature, use top-k sampling |
| Slow generation | High latency | Use KV-caching, reduce beam width |
| OOV tokens | Unknown words | Use subword tokenization (BPE) |

## Applications

### Content Creation
- **Article Writing**: Generate blog posts, news
- **Story Generation**: Creative fiction
- **Poetry**: Rhyming and structured text
- **Code Generation**: Programming assistance

### Conversational AI
- **Chatbots**: Customer service, virtual assistants
- **Dialogue Systems**: Task-oriented conversations
- **Personality-Based Bots**: Consistent character

### Translation & Transformation
- **Machine Translation**: Language to language
- **Summarization**: Long text to short
- **Paraphrasing**: Rephrase content
- **Style Transfer**: Formal ↔ Informal

### Code & Data
- **SQL Generation**: Natural language to queries
- **Code Completion**: Programming assistance
- **Data Augmentation**: Synthetic text generation

## Evaluation Metrics

### Automatic Metrics

- **Perplexity**: Model confidence (lower is better)
  ```
  Perplexity = exp(cross_entropy_loss)
  ```
- **BLEU**: N-gram overlap (translation, 0-100)
- **ROUGE**: Recall of n-grams (summarization)
- **METEOR**: Semantic matching

### Human Evaluation

- **Fluency**: Grammatical correctness
- **Coherence**: Logical flow
- **Relevance**: Topic adherence
- **Diversity**: Variety of outputs

## Pre-trained Models

### GPT Family (OpenAI)
- GPT-2: 117M to 1.5B parameters
- GPT-3: 175B parameters
- GPT-4: Multimodal capabilities

### Others
- **BERT**: Bidirectional encoder (not generative)
- **T5**: Text-to-text framework
- **BART**: Denoising autoencoder
- **XLNet**: Permutation language modeling

## Fine-Tuning Example

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare dataset
train_dataset = load_custom_dataset('train.txt')

# Fine-tune
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args
)
trainer.train()

# Generate
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=100, do_sample=True)
generated_text = tokenizer.decode(output[0])
```

## Advanced Topics

- **Prompt Engineering**: Crafting effective prompts
- **Few-Shot Learning**: Learn from examples
- **Instruction Tuning**: Follow natural language instructions
- **RLHF**: Reinforcement Learning from Human Feedback
- **Constitutional AI**: Alignment and safety

## Next Steps

After mastering Generative NLP:
- **Chapter 16**: Introduction to Large Language Models
- **Chapter 17**: GPT Architecture Deep Dive
- **Chapter 18**: LangChain Applications

## Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Text Generation Tutorial - Hugging Face](https://huggingface.co/blog/how-to-generate)

---

**Start Generating Text!** Launch the Streamlit apps and create your own language models!
