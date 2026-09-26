from transformers import AutoTokenizer, AutoModel
import torch


def main():

    # 1. Load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    # 2. Text to process
    text = "I am learning transformers."

    # 3. Convert text into tokens
    tokens = tokenizer.tokenize(text)

    print("Tokens:")
    print(tokens)

    # 4. Convert tokens into token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("\nToken IDs:")
    print(token_ids)

    # 5. Encode the text into tensors
    encoded = tokenizer(
        text,
        return_tensors="pt"
    )

    print("\nEncoded:")
    print(encoded)

    print("\nInput IDs:")
    print(encoded["input_ids"])

    print("\nAttention Mask:")
    print(encoded["attention_mask"])

    # 6. Load the pretrained BERT model
    model = AutoModel.from_pretrained(
        "bert-base-uncased"
    )

    # 7. Pass the encoded text through the model
    with torch.no_grad():

        outputs = model(**encoded)

    # 8. Inspect the model output
    print("\nLast Hidden State Shape:")
    print(outputs.last_hidden_state.shape)


if __name__ == "__main__":
    main()