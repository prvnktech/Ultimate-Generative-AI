from openai import OpenAI
# Initialize the OpenAI client
client = OpenAI(api_key="your_api_key_here")
# Define the prompt
prompt = (
    "Explain how to fine-tune a language model for legal text summarization."
)
# Generate response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=100,
    temperature=0.4
)
# Print the response
print(response.choices[0].message.content)