from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load BERT with attention outputs
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
)

# Input sentence
text = "The cat sat on the mat."

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Get tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("Tokens:")
print(tokens)


# Run the model
with torch.no_grad():
    outputs = model(**inputs)


# Get attention weights
attentions = outputs.attentions

print("\nNumber of layers:")
print(len(attentions))

print("\nShape of attention from first layer:")
print(attentions[0].shape)

# First layer
layer = attentions[0]

# First attention head
head = layer[0, 0]

print("\nFirst attention head:")
print(head)

print("\nAttention matrix shape:")
print(head.shape)

plt.figure(figsize=(10, 8))

plt.imshow(head)

plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)

plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title("BERT Attention — Layer 1, Head 1")

plt.colorbar()

plt.tight_layout()
plt.show()