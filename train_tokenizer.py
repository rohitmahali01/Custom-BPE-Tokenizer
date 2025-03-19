# -*- coding: utf-8 -*-
"""Custom BPE Tokenizer for Any Language"""

# Install necessary libraries
!pip install datasets transformers tokenizers

from google.colab import files
import os
import glob
from tqdm.auto import tqdm
from tokenizers import ByteLevelBPETokenizer

# Upload the text file
uploaded = files.upload()

# Get the uploaded file name dynamically
file_name = list(uploaded.keys())[0]

# Read and preview the first 500 characters
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

print("First 500 characters:\n", text[:500])

# Read the file contents line by line
with open(file_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Initialize variables for batch processing , tailor it using your custom batch_size
text_data = []
file_count = 0
batch_size = 5000  # Save every 5000 lines

# Process each line from the file
for line in tqdm(lines, desc="Processing lines"):
    text = line.strip()
    text_data.append(text)

    # Save every 5,000 lines as separate files
    if len(text_data) >= batch_size:
        file_path = f'text_{file_count}.txt'
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(text_data))

        text_data = []  # Clear the list for the next batch
        file_count += 1

# Save any remaining text
if text_data:
    file_path = f'text_{file_count}.txt'
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(text_data))

print(f"Splitting complete! {file_count + 1} files saved.")

# Get all text files dynamically
paths = glob.glob("text_*.txt")  # Finds all saved text files (e.g., text_0.txt, text_1.txt)

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the uploaded text files also Train tokenizer with **language-specific optimizations**
tokenizer.train(files=paths, vocab_size=30000, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])   #

# Define tokenizer directory
tokenizer_dir = "custom_tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

# Save the trained tokenizer
tokenizer.save_model(tokenizer_dir)

print("Tokenizer training complete! Files saved in 'custom_tokenizer/' directory.")
