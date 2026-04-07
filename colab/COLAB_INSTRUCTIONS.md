# Google Colab FAISS & Embedding Pipeline

**Why `all-MiniLM-L6-v2`?**  
This model is chosen because it produces high-quality sentence embeddings but is incredibly small (~80MB) and fast. It generates 384-dimensional vectors, meaning we save memory, making it the perfect candidate for local inference on a CPU without requiring a GPU.

---

### Step-by-Step Instructions

1. Open [Google Colab](https://colab.research.google.com/) and create a new **blank notebook**.
2. No GPU required! Leave the runtime as standard CPU.
3. Copy and paste each of the blocks below into their own cell, and run them top-to-bottom.

#### Cell 1: Install Dependencies
```python
!pip install -q sentence-transformers faiss-cpu datasets pandas numpy
```

#### Cell 2: Load Model
```python
from sentence_transformers import SentenceTransformer

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
print(f"Model loaded: {model_name}")
```

#### Cell 3: Load Dataset (10k-50k docs)
```python
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
dataset = load_dataset("ag_news", split="train[:25000]")
df = pd.DataFrame(dataset)

df['text'] = df['text'].str.strip()
docs = df['text'].tolist()
print(f"Loaded {len(docs)} documents.")
df.head()
```

#### Cell 4: Generate Embeddings (Batching used for speed)
```python
import numpy as np
import time

print("Starting embedding generation...")
start_time = time.time()

embeddings = model.encode(docs, batch_size=256, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32') # Critical for FAISS

end_time = time.time()
print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
print(f"Embeddings shape: {embeddings.shape}")
```

#### Cell 5: Build FAISS Index
```python
import faiss

dimension = embeddings.shape[1] 

# IndexFlatL2 gives exact closest-neighbor searches. Perfect for < 50k documents.
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built. Total vectors: {index.ntotal}")
```

#### Cell 6: Save Files
```python
import pickle
import os

os.makedirs('output', exist_ok=True)

faiss.write_index(index, "output/faiss.index")
np.save("output/embeddings.npy", embeddings)

metadata = df.to_dict('records')
with open('output/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Saved files in 'output/'.")
```

#### Cell 7: Download to your PC
```python
from google.colab import files

print("Downloading artifacts...")
files.download("output/faiss.index")
files.download("output/embeddings.npy")
files.download("output/metadata.pkl")
```

---

### What's Next?
After you run these cells, your browser will download three files:
*   `faiss.index`
*   `embeddings.npy`
*   `metadata.pkl`

Once downloaded, please place them in the `rag_semantic_search/models/` folder we will create soon. Let me know when you have these downloaded!
