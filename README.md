# Transformer Attention From Scratch

A learning project implementing Transformer attention from scratch using PyTorch.

## Goals

- Understand Query, Key, and Value
- Implement scaled dot-product attention
- Understand self-attention
- Visualize attention weights
- Explore Transformer architecture
- Later use Hugging Face Transformers

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter
- Hugging Face Transformers



## Day 3 - Token Embeddings

Today I connected text representations to the attention mechanism.

### What I implemented

- Created a small vocabulary
- Converted tokens into integer IDs
- Implemented a PyTorch embedding layer
- Converted token IDs into vector representations
- Used embeddings as Q, K and V for a simplified attention experiment
- Verified attention weight dimensions and row sums

### Pipeline

Text
↓
Token IDs
↓
Token Embeddings
↓
Q, K, V
↓
Scaled Dot-Product Attention
↓
Context-Aware Representations

### Key Concepts

An embedding converts each token ID into a dense vector representation.

Attention then combines information from different token representations to produce context-aware outputs.

### Example

Input:

"The cat sat on the mat"

Token IDs:

[0, 1, 2, 3, 0, 4]

The resulting embedding matrix has shape:

(6, embedding_dimension)

The attention matrix has shape:

(6, 6), since every token can attend to every token.


## Day 4 - Scaled Dot-Product Attention

Implemented scaled dot-product attention from scratch using PyTorch.

The implementation follows:

Attention(Q,K,V) =
softmax(QK^T / sqrt(d_k))V

Steps implemented:

1. Calculate QK^T
2. Scale scores by sqrt(d_k)
3. Apply softmax
4. Multiply attention weights by V
5. Generate the final attention output

### Key Concepts

- Query-Key similarity
- Attention scores
- Scaling
- Softmax
- Attention weights
- Weighted value aggregation