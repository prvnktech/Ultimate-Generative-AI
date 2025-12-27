import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# 1. Load pre-trained model and tokenizer
# 'bert-base-uncased' is a standard starting point for English text
model_name = "bert-base-uncased"
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 2. Prepare training data
texts = ["This movie was fantastic!", "I did not like this film."]
labels = [1, 0]  # 1 = Positive, 0 = Negative

# Tokenize inputs (converts text to input_ids, attention_masks, etc.)
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

# Convert labels to a TensorFlow tensor
targets = tf.convert_to_tensor(labels)

# 3. Fine-tuning setup
# Using a very small learning rate as is standard for fine-tuning Transformers
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# 4. Training
print("Starting training...")
history = model.fit(
    inputs.data, 
    targets, 
    epochs=3, 
    batch_size=2
)

# 5. Example Inference (Post-Training)
test_text = ["I really enjoyed the acting in this."]
test_inputs = tokenizer(test_text, padding=True, truncation=True, return_tensors="tf")
predictions = model.predict(test_inputs.data)
probabilities = tf.nn.softmax(predictions.logits, axis=-1)
print(f"Prediction probabilities (Neg, Pos): {probabilities.numpy()}")