import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

st.title('Semantic Search App')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to load pre-computed embeddings
@st.cache_resource
def load_embeddings():
    embeddings = np.load('embeddings.npy')
    return embeddings

# Function to load sentences
@st.cache_resource
def load_sentences():
    with open('sentences.txt', 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    return sentences

# Function to load the embedding model
@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    return model

# Load resources
embeddings = load_embeddings()
sentences = load_sentences()
model = load_model()

# User input
user_input = st.text_input('Enter a job title:')

if user_input:
    # Compute embedding for the user input
    input_embedding = model.encode([user_input], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Calculate cosine similarities
    similarities = np.inner(input_embedding, embeddings)

    # Get indices of top 5 matches
    top5_indices = similarities.argsort()[-5:][::-1]

    # Display top 5 matching sentences
    st.subheader('Top 5 Matches:')
    for idx in top5_indices:
        st.write(f"**Job Posting:** {sentences[idx]}")
        st.write(f"**Similarity Score:** {similarities[idx]:.4f}")
        st.write('---')