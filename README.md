# Custom-BPE-Tokenizer
# A custom tokenizer creation tool , leveraging Byte-Pair Encoding (BPE)

This repository provides a Byte-Pair Encoding (BPE) tokenizer that can be trained on any language dataset. it enables efficient tokenization and can be easily adapted for different scripts and languages.

# Features
* Custom BPE tokenization for any language.
* Fast and optimized using Hugging Face’s tokenizers.
* Scalable and adaptable for different linguistic datasets.
* Easy integration into NLP pipelines.

# Example Usage
This tokenizer was used to create a Santali (Olchiki script) tokenizer. Check out the implementation here: (https://github.com/rohitmahali01/Santali-Tokenizer/tree/main)

# Working Pipeline
* Step 1 - Prepare Your Text Corpus
- Collect a large dataset of text in your target language.
Ensure the dataset is UTF-8 encoded to correctly handle special characters and unique scripts.
The corpus should represent diverse and natural language usage for better tokenization.
* Step 2 - Train the Tokenizer
- Run train_tokenizer.py to train a Byte Pair Encoding (BPE) tokenizer on your dataset.
The script will generate:
vocab.json – Vocabulary file
merges.txt – Merge rules for BPE
These files will be saved inside a model directory (e.g., custom_tokenizer/).
* Step 3 - Test the Tokenizer
Run test_tokenizer.py to evaluate how well the trained tokenizer works.
- Input sample text in your target language and verify:
Tokenized output
Token IDs
Whether decoding back reconstructs the original text accurately.

# Documentations are still being updated.....
# This tool will always be in development to make it adhere to modern optimized frameworks . 
