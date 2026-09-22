
import torch

from attention import scaled_dot_product_attention


# 1. Create vocabulary


vocab = {
    "the": 0,
    "cat": 1,
    "sat": 2,
    "on": 3,
    "mat": 4
}


# -----------------------------
# 2. Tokenize sentence
# -----------------------------

sentence = ["the", "cat", "sat", "on", "the", "mat"]

#convert words into int token ids
token_ids = [vocab[word] for word in sentence]

print("Sentence:")
print(sentence)

print("\nToken IDs:")
print(token_ids)


# -----------------------------
# 3. Convert to tensor
# -----------------------------

token_tensor = torch.tensor(token_ids)

print("\nToken Tensor:")
print(token_tensor)


# -----------------------------
# 4. Create embedding layer
# -----------------------------

embedding_dim = 8

embedding = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=embedding_dim
)


# -----------------------------
# 5. Generate embeddings
# -----------------------------

embedded = embedding(token_tensor)

print("\nEmbeddings:")
print(embedded)

print("\nEmbedding Shape:")
print(embedded.shape)


# -----------------------------
# 6. Use embeddings as Q, K, V
# -----------------------------

Q = embedded
K = embedded
V = embedded


# -----------------------------
# 7. Apply attention
# -----------------------------

output, weights = scaled_dot_product_attention(Q, K, V)


print("\nAttention Weights:")
print(weights)
print("\nRow sums:")
print(weights.sum(dim=-1))

print("\nAttention Output:")
print(output)

print("\nAttention Output Shape:")
print(output.shape)

print("\nOriginal embedding of 'the':")
print(embedded[0])

print("\nContextual output for first 'the':")
print(output[0])
