import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class SemanticIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2', models_dir='../models'):
        print(f"Loading Sentence-Transformer model: {model_name}...")
        # Keeping model locally resident in memory, so inference is blazing fast
        self.model = SentenceTransformer(model_name)
        
        print("Loading FAISS index & metadata...") # Load pre-computed artifacts
        self.index = faiss.read_index(os.path.join(models_dir, 'faiss.index'))
        
        with open(os.path.join(models_dir, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
            
        print(f"Loaded successfully! Index contains {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5):
        # 1. Embed incoming query on-the-fly (Fast local CPU operation)
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32') # Needs to match FAISS req
        
        # 2. Search exact L2 distance
        D, I = self.index.search(query_embedding, top_k)
        
        # 3. Retrieve metadata using the returned index
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.metadata):
                doc_meta = self.metadata[idx]
                results.append({
                    "id": int(idx),
                    "score": float(dist), # FAISS L2 distance - closer to 0 is better!
                    "text": doc_meta.get('text', ''),
                    "label": doc_meta.get('label', -1)
                })
        return results
