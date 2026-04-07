"""
Colab Pipeline: Embedding Generation and FAISS Indexing
This script is intended to be run in Google Colab cell by cell.
Please refer to COLAB_INSTRUCTIONS.md for the cell-by-cell breakdown.
"""

# ==========================================
# CELL 1: Install Dependencies
# ==========================================
# !pip install -q sentence-transformers faiss-cpu datasets pandas numpy

# ==========================================
# CELL 2: Load Model
# ==========================================
from sentence_transformers import SentenceTransformer

# We use all-MiniLM-L6-v2. It is highly optimized for performance and size (~80MB),
# runs extremely fast on CPU, and gives robust contextual embeddings for search.
# Output dimension is 384.
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
print(f"Model loaded: {model_name}")

# ==========================================
# CELL 3: Load Dataset
# ==========================================
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
# Using AG News dataset, slicing the first 25,000 documents to fit the 10k-50k constraint perfectly.
dataset = load_dataset("ag_news", split="train[:25000]")
df = pd.DataFrame(dataset)

# The dataset has 'text' and 'label'. We strip any leading/trailing whitespace.
df['text'] = df['text'].str.strip()
docs = df['text'].tolist()
print(f"Loaded {len(docs)} documents.")
df.head()

# ==========================================
# CELL 4: Generate Embeddings
# ==========================================
import numpy as np
import time

print("Starting embedding generation (Expected to take ~2-5 minutes on Colab CPU)...")
start_time = time.time()

# Batching to speed up inference and manage memory cleanly
embeddings = model.encode(docs, batch_size=256, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32') # FAISS requires exact float32 types

end_time = time.time()
print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
print(f"Embeddings shape: {embeddings.shape}")

# ==========================================
# CELL 5: Indexing with FAISS
# ==========================================
import faiss

# The dimensionality must match the embedding size (384)
dimension = embeddings.shape[1] 

# IndexFlatL2 performs exact Euclidean distance search. 
# For < 50k documents, this is entirely sufficient and extremely fast.
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built successfully. Total vectors in index: {index.ntotal}")

# ==========================================
# CELL 6: Save Artifacts
# ==========================================
import pickle
import os

os.makedirs('output', exist_ok=True)

# 1. Save FAISS index
faiss.write_index(index, "output/faiss.index")

# 2. Save raw embeddings as requested
np.save("output/embeddings.npy", embeddings)

# 3. Save metadata mapping (so we can map indexes back to actual text locally)
metadata = df.to_dict('records')
with open('output/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Saved faiss.index, embeddings.npy, and metadata.pkl securely in 'output/' directory.")

# ==========================================
# CELL 7: Download to Local Machine
# ==========================================
# from google.colab import files
# print("Downloading artifacts to your machine...")
# files.download("output/faiss.index")
# files.download("output/embeddings.npy")
# files.download("output/metadata.pkl")
