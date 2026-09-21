import torch
import math


def scaled_dot_product_attention(Q, K, V):

    dk = K.size(-1)

#calculate the simialrity between query and keys
    scores = Q @ K.T

#scale scores to prevent extreme values and improve numerical stability
    scaled_scores = scores / math.sqrt(dk)

#convert scores to attention weights
    attention_weights = torch.softmax(scaled_scores, dim=-1)

#create weighted combinations of values based on attention weights
    output = attention_weights @ V

    return output, attention_weights


Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
    ])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
    ])

V = torch.tensor([
    [100.0, 0.0],
    [0.0, 20.0]
    ])  

output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention weights :")
print(weights)

print("\nAttention output:")
print(output)





    