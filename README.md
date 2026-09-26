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

## Day 5 - Multi-Head Attention

Implemented Multi-Head Attention from scratch using PyTorch.

### Concepts learned

- Multiple attention heads
- Q, K, V projections
- Splitting embeddings into attention heads
- Scaled dot-product attention
- Concatenating attention heads
- Output projection
- Tensor shape manipulation

### Example

d_model = 8
num_heads = 2
head_dim = 4

Input shape:
(1, 4, 8)

Attention weights:
(1, 2, 4, 4)

Output shape:
(1, 4, 8)
**## Day 6 - Positional Encoding + Hugging Face Introduction**

Today I learned how Transformers handle token position and explored pretrained Transformer models using Hugging Face.

**### Concepts learned**

- Why positional information is needed in Transformers

- Sinusoidal positional encoding

- Adding positional encoding to token embeddings

- Hugging Face Transformers library

- Tokenization using `AutoTokenizer`

- Token IDs and attention masks

- Loading pretrained BERT using `AutoModel`

- Understanding Transformer hidden states

**### Positional Encoding**

Implemented sinusoidal positional encoding from scratch using PyTorch.

Positional encoding adds information about the position of each token because self-attention itself does not inherently know token order.

The positional encoding is added to the token embeddings:

```text
Token Embeddings
       +
Positional Encoding
       ↓
Position-Aware Representations
```

**### Hugging Face**

Used Hugging Face Transformers to tokenize text and load a pretrained BERT model.

Example:

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Token IDs
 ↓
BERT
 ↓
Hidden States
```

**### Example**

Input:

`"I am learning transformers."`

BERT tokenizer produces subword tokens such as:

```text
['i', 'am', 'learning', 'trans', '##of', '##rm', '##ers', '.']
```

The tokenizer then converts these tokens into numerical token IDs.

The encoded input also contains an attention mask indicating which tokens should be considered by the model.

**### Key Connection**

The from-scratch implementation helps understand how attention works internally, while Hugging Face provides pretrained Transformer models that can be used directly for real-world NLP tasks.
