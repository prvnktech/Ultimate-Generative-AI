import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def run_gpt2_inference():
    # 1. Select Device (Uses GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load pre-trained model and tokenizer
    model_name = "gpt2"
    print("Loading model... (this may take a minute the first time)")
    
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # GPT-2 needs a padding token defined to handle batching properly
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Prepare Input
    input_text = "How can AI models be fine-tuned for domain-specific tasks?"
    
    # We return tensors for PyTorch ('pt') and move them to the same device as the model
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 4. Generate Output
    print("Generating text...")
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=100,            # Limits the length of the response
        num_return_sequences=1,    # Number of different sentences to generate
        no_repeat_ngram_size=2,    # Prevents the model from repeating phrases
        do_sample=True,            # Allows for more creative/varied responses
        top_k=50,                  # Limits "vocabulary" to top 50 likely next words
        top_p=0.95,                # Cumulative probability for word selection
        temperature=0.7,           # 1.0 is default, lower is more focused/deterministic
        pad_token_id=tokenizer.eos_token_id
    )

    # 5. Decode and Print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*30)
    print("PROMPT:", input_text)
    print("="*30)
    print("AI RESPONSE:\n", response)
    print("="*30)

if __name__ == "__main__":
    run_gpt2_inference()