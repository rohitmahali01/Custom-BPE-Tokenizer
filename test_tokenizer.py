# Load the trained tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=f"{tokenizer_dir}/vocab.json",
    merges_filename=f"{tokenizer_dir}/merges.txt"
)

# Example text (Replace with any language text)
example_text = "Sample text for tokenization"

# Encode the text
output = tokenizer.encode(example_text)

# Print tokenized output
print("Tokens:", output.tokens)
print("Token IDs:", output.ids)

# Decode the tokens
decoded_text = tokenizer.decode(output.ids)
print("Decoded Text:", decoded_text)
