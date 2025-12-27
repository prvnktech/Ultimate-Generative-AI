import streamlit as st
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os
import tempfile
from pathlib import Path
import json

st.set_page_config(page_title="LLM Fine-tuning Dashboard", layout="wide")

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = "Not Started"
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

st.title("🤖 LLM Fine-tuning Dashboard")
st.markdown("Fine-tune language models on your custom dataset")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Base Model",
    [
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-hf",
        "gpt2",
        "gpt2-medium",
        "EleutherAI/pythia-1.4b"
    ]
)

# Training parameters
st.sidebar.subheader("Training Parameters")
num_epochs = st.sidebar.slider("Number of Epochs", 1, 10, 3)
batch_size = st.sidebar.slider("Batch Size", 1, 8, 1)
gradient_accumulation = st.sidebar.slider("Gradient Accumulation Steps", 1, 16, 8)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    value=2e-5,
    format_func=lambda x: f"{x:.0e}"
)
max_length = st.sidebar.slider("Max Sequence Length", 128, 1024, 512, step=128)
warmup_steps = st.sidebar.number_input("Warmup Steps", 0, 500, 100)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "🔧 Training", "💬 Test Model", "📊 Settings"])

# Tab 1: Data Upload
with tab1:
    st.header("Upload Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data")
        train_file = st.file_uploader(
            "Upload training text file",
            type=['txt'],
            key='train'
        )
        if train_file:
            train_content = train_file.read().decode('utf-8')
            st.success(f"✅ Loaded {len(train_content.split())} words")
            with st.expander("Preview Training Data"):
                st.text(train_content[:500] + "...")
    
    with col2:
        st.subheader("Validation Data")
        val_file = st.file_uploader(
            "Upload validation text file",
            type=['txt'],
            key='val'
        )
        if val_file:
            val_content = val_file.read().decode('utf-8')
            st.success(f"✅ Loaded {len(val_content.split())} words")
            with st.expander("Preview Validation Data"):
                st.text(val_content[:500] + "...")
    
    st.info("💡 Tip: Your text files should contain one example per line or paragraphs separated by blank lines.")

# Tab 2: Training
with tab2:
    st.header("Model Training")
    
    # Display current configuration
    with st.expander("📋 Current Configuration", expanded=True):
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.markdown(f"""
            - **Model**: {model_name}
            - **Epochs**: {num_epochs}
            - **Batch Size**: {batch_size}
            - **Gradient Accumulation**: {gradient_accumulation}
            """)
        with config_col2:
            st.markdown(f"""
            - **Learning Rate**: {learning_rate:.0e}
            - **Max Length**: {max_length}
            - **Warmup Steps**: {warmup_steps}
            - **Effective Batch Size**: {batch_size * gradient_accumulation}
            """)
    
    # Check if data is uploaded
    if train_file and val_file:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.training_status = "Running"
            
            try:
                with st.spinner("Initializing training..."):
                    # Create temporary files
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        train_path = Path(tmp_dir) / "train.txt"
                        val_path = Path(tmp_dir) / "valid.txt"
                        
                        # Write uploaded files
                        with open(train_path, 'w', encoding='utf-8') as f:
                            f.write(train_content)
                        with open(val_path, 'w', encoding='utf-8') as f:
                            f.write(val_content)
                        
                        # Status updates
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        status_placeholder.info("📦 Loading tokenizer and model...")
                        
                        # Load tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        progress_bar.progress(20)
                        
                        # Load model
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        progress_bar.progress(40)
                        status_placeholder.info("📊 Loading and tokenizing dataset...")
                        
                        # Load dataset
                        dataset = load_dataset(
                            "text",
                            data_files={
                                "train": str(train_path),
                                "validation": str(val_path)
                            }
                        )
                        
                        # Tokenization function
                        def tokenize_fn(batch):
                            return tokenizer(
                                batch["text"],
                                truncation=True,
                                padding="max_length",
                                max_length=max_length
                            )
                        
                        # Tokenize
                        tokenized_ds = dataset.map(
                            tokenize_fn,
                            batched=True,
                            remove_columns=["text"]
                        )
                        
                        progress_bar.progress(60)
                        status_placeholder.info("🔧 Setting up training...")
                        
                        # Data collator
                        data_collator = DataCollatorForLanguageModeling(
                            tokenizer=tokenizer,
                            mlm=False
                        )
                        
                        # Training arguments
                        training_args = TrainingArguments(
                            output_dir="./mistral-finetuned",
                            num_train_epochs=num_epochs,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            gradient_accumulation_steps=gradient_accumulation,
                            evaluation_strategy="epoch",
                            save_strategy="epoch",
                            logging_steps=50,
                            fp16=torch.cuda.is_available(),
                            learning_rate=learning_rate,
                            warmup_steps=warmup_steps,
                            save_total_limit=2,
                            report_to="none"
                        )
                        
                        # Create trainer
                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=tokenized_ds["train"],
                            eval_dataset=tokenized_ds["validation"],
                            data_collator=data_collator
                        )
                        
                        progress_bar.progress(80)
                        status_placeholder.info("🏋️ Training in progress...")
                        
                        # Train
                        trainer.train()
                        
                        progress_bar.progress(100)
                        status_placeholder.success("✅ Training completed!")
                        
                        # Save model
                        model.save_pretrained("./mistral-finetuned")
                        tokenizer.save_pretrained("./mistral-finetuned")
                        
                        st.session_state.training_status = "Completed"
                        st.session_state.model_loaded = True
                        st.balloons()
                        
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.session_state.training_status = "Failed"
    else:
        st.warning("⚠️ Please upload both training and validation data files in the 'Data Upload' tab.")
    
    # Training status
    st.divider()
    st.subheader("Training Status")
    if st.session_state.training_status == "Not Started":
        st.info("⏳ Training not started")
    elif st.session_state.training_status == "Running":
        st.warning("🏃 Training in progress...")
    elif st.session_state.training_status == "Completed":
        st.success("✅ Training completed successfully!")
    elif st.session_state.training_status == "Failed":
        st.error("❌ Training failed")

# Tab 3: Test Model
with tab3:
    st.header("Test Your Fine-tuned Model")
    
    if st.session_state.model_loaded or os.path.exists("./mistral-finetuned"):
        test_prompt = st.text_area(
            "Enter a prompt to test the model:",
            placeholder="Once upon a time...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_new_tokens = st.slider("Max New Tokens", 10, 500, 100)
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
        
        if st.button("Generate", type="primary"):
            if test_prompt:
                with st.spinner("Generating..."):
                    try:
                        # Load fine-tuned model
                        tokenizer = AutoTokenizer.from_pretrained("./mistral-finetuned")
                        model = AutoModelForCausalLM.from_pretrained(
                            "./mistral-finetuned",
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        # Generate
                        inputs = tokenizer(test_prompt, return_tensors="pt")
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True
                        )
                        
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        st.subheader("Generated Output:")
                        st.write(generated_text)
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
            else:
                st.warning("Please enter a prompt")
    else:
        st.info("⚠️ No fine-tuned model found. Please train a model first.")

# Tab 4: Settings
with tab4:
    st.header("Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Information")
        st.write(f"PyTorch Version: {torch.__version__}")
        st.write(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            st.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    with col2:
        st.subheader("Model Management")
        if st.button("🗑️ Clear Fine-tuned Model"):
            if os.path.exists("./mistral-finetuned"):
                import shutil
                shutil.rmtree("./mistral-finetuned")
                st.session_state.model_loaded = False
                st.success("Model cleared successfully!")
                st.rerun()
            else:
                st.info("No model to clear")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Make sure you have sufficient GPU memory for training large models</p>
</div>
""", unsafe_allow_html=True)