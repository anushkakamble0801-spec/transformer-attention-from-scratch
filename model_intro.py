from transformers import AutoTokenizer, AutoModel
import torch


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load pretrained model
model = AutoModel.from_pretrained("bert-base-uncased")

# Input text
text = "The cat sat on the mat."

# Tokenize text
inputs = tokenizer(text, return_tensors="pt")

print("Input IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])


# Pass input through the Transformer
with torch.no_grad():
    outputs = model(**inputs)

print("\nLast Hidden State Shape:")
print(outputs.last_hidden_state.shape)

print("\nFirst Token Representation:")
print(outputs.last_hidden_state[0, 0])

print("\nRepresentation Shape:")
print(outputs.last_hidden_state[0, 0].shape)