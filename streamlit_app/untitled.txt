import os
import sys

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
from tqdm import trange

# Define paths
build_project_path = r"C:\Users\L110006\OneDrive - Eli Lilly and Company\personal files\build project\fine-tuning-build-project"
streamlit_app_data_path = os.path.join(build_project_path, "streamlit_app", "data")

# Detect device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f'Using device: {device}')

# Load job postings
job_postings_df = pd.read_parquet(os.path.join(streamlit_app_data_path, 'job_postings.parquet'))
job_titles = job_postings_df['job_posting_title'].to_list()

# Load models
default_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
fine_tuned_model_path = os.path.join(streamlit_app_data_path, 'fine_tuned_model')
fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)

# Compute embeddings in chunks
default_embeddings = []
fine_tuned_embeddings = []
for i in trange(0, len(job_titles), 100):
    chunk = job_titles[i:i+100]
    default_embeddings.append(default_model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True))
    fine_tuned_embeddings.append(fine_tuned_model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True))

default_embeddings = np.concatenate(default_embeddings)
print(default_embeddings.shape)

fine_tuned_embeddings = np.concatenate(fine_tuned_embeddings)
print(fine_tuned_embeddings.shape)

# Save embeddings
np.save(os.path.join(streamlit_app_data_path, 'default_embeddings.npy'), default_embeddings)
np.save(os.path.join(streamlit_app_data_path, 'fine_tuned_embeddings.npy'), fine_tuned_embeddings)
