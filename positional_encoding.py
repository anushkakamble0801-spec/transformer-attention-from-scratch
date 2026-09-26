import torch
import math


def positional_encoding(seq_len, d_model):

    pe = torch.zeros(seq_len, d_model)

    for pos in range(seq_len):

        for i in range(0, d_model, 2):

            pe[pos, i] = math.sin(
                pos / (10000 ** (i / d_model))
            )

            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** (i / d_model))
                )

    return pe


if __name__ == "__main__":

    seq_len = 6
    d_model = 8

    pe = positional_encoding(
        seq_len,
        d_model 
    )

    print("Positional Encoding:")
    print(pe)

    print("\nShape:")
    print(pe.shape)




            
                
                
            
