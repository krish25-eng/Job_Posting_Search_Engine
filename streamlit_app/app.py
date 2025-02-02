import os
import sys

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

st.title('Job Posting Search Engine')
device = get_device()

# Function to load pre-computed embeddings
@st.cache_resource
def load_embeddings():
    embeddings = np.load('fine_tuned_embeddings.npy')
    return embeddings

# Function to load sentences
@st.cache_resource
def load_job_postings():
    job_postings_df = pd.read_parquet('data/job_postings.parquet')
    job_postings_df['posting'] = job_postings_df['job_posting_title'] + ' @ ' + job_postings_df['company']
    return job_postings_df['posting'].to_list()


# Function to load the embedding model
@st.cache_resource
def load_model():
    fine_tuned_model_path = os.path.join('data', 'fine_tuned_model')
    fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)
    return fine_tuned_model

# Load resources
embeddings = torch.tensor(load_embeddings()[:1000], device=device)
job_postings = load_job_postings()[:1000]
model = load_model()

# User input
user_input = st.text_input('Enter a job title:')

if user_input:
    # Compute embedding for the user input
    with torch.inference_mode():
        input_embedding = model.encode([user_input], normalize_embeddings=True, convert_to_tensor=True)[0]

        # Calculate cosine similarities
        similarities = torch.inner(input_embedding, embeddings)

        # Grab the top 5 most similar job posting titles
        top5_indices = torch.argsort(similarities, descending=True)[:10]


    # Display top 5 matching postings
    st.subheader('Top 10 Matches:')
    for idx in top5_indices:
        st.write(f"**Job Posting:** {job_postings[idx]}")
        st.write(f"**Similarity Score:** {similarities[idx]:.4f}")
        st.write('---')