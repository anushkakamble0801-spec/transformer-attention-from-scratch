import torch

# -------------------------
# 1. Define Q, K and V
# -------------------------

Q = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])

K = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# -------------------------
# 2. Calculate QK^T
# -------------------------

scores = Q @ K.T

print("\nQK^T:")
print(scores)


# -------------------------
# 3. Scale scores
# -------------------------

d_k = K.shape[-1]

scaled_scores = scores / torch.sqrt(
    torch.tensor(d_k, dtype=torch.float32)
)

print("\nScaled scores:")
print(scaled_scores)


# -------------------------
# 4. Softmax
# -------------------------

attention_weights = torch.softmax(
    scaled_scores,
    dim=-1
)

print("\nAttention weights:")
print(attention_weights)


# -------------------------
# 5. Calculate attention output
# -------------------------

output = attention_weights @ V

print("\nAttention output:")
print(output)