from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Sample sentences (replace this with your actual data)
sentences = [
    "The cat sits on the mat.",
    "Dogs are man's best friend.",
    "Artificial intelligence is the future.",
    "I love reading books on machine learning.",
    "Streamlit makes it easy to build web apps."
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Compute embeddings
embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)

# Save embeddings and sentences
np.save('embeddings.npy', embeddings)
with open('sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in sentences:
        f.write(f"{sentence}\n")