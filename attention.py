import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        output = output.transpose(
            1, 2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.W_o(output)

        return output, attention_weights


if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    x = torch.randn(
        batch_size,
        seq_len,
        d_model
    )

    model = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    output, attention_weights = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)

    print("\nAttention weight sums:")
    print(attention_weights.sum(dim=-1))

    print("\nHead 1 attention:")
    print(attention_weights[0, 0])

    print("\nHead 2 attention:")
    print(attention_weights[0, 1])