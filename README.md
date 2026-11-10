# Embedding-Based Semantic Search System (RAG)

A production-ready Semantic Search engine that converts massive local document datasets into rapidly searchable vector representations (Embeddings) using FAISS. The system completely decouples the heavy indexing process from the serving layer, achieving high scalability with zero GPU dependencies.

## 🌟 Features
* **Lightning-Fast Vector Search:** Index up to millions of texts using exact L2-distance FAISS search, fetching matches in < 100 milliseconds locally.
* **GPU-Free Local Inference:** Utilizes `all-MiniLM-L6-v2`, an incredibly lightweight mathematical embedding model optimized for CPU inference without bottlenecking.
* **Retrieval-Augmented Generation (RAG):** Includes an integrated Groq API adapter capable of taking raw fetched documents and passing them dynamically into a Llama-3 LLM to synthesize cited natural-language answers.
* **Beautiful User Interface:** A stunning Streamlit modern frontend tailored for a high-end, premium user experience.

## 🏗️ Architecture Design

```mermaid
graph TD
    A[Raw Document Dataset] -->|Batched| B[SentenceTransformers Colab Pipeline]
    B -->|Offline Generation| C(Embeddings .npy)
    B -->|Offline Generation| D(FAISS .index)
    B -->|Offline Generation| E(Metadata .pkl)
    
    C --> F[FastAPI Local Server]
    D --> F
    E --> F
    
    G[Streamlit Frontend] -->|HTTP POST Request| F
    F -->|Return Top-K matches| H[Retrieved Semantic Documents]
    H -.->|Optional Context Injection| I[Groq Llama-3 Generator]
    I -.-> G
```

## 🚀 Setup & Execution

### 1. Build the vector index
The index files are not stored in git. Build them once:

```bash
cd backend/
pip install -r requirements.txt
python build_index.py                  # bundled 36-document sample corpus (a few seconds)
python build_index.py --ag-news 25000  # full AG News demo (needs `pip install datasets`)
python build_index.py --file docs.txt  # your own corpus, one document per line
```

For very large corpora you can instead run the notebook-style pipeline in `colab/` on a free
Colab instance and copy `faiss.index`, `embeddings.npy` and `metadata.pkl` into `models/`.

### 2. Start the Backend API (FastAPI)
The backend loads the entire vector database into RAM.
```bash
cd backend/
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*(Optional RAG)*: Before launching, set your API key environment variable in Powershell:
`$env:GROQ_API_KEY="gsk_..."`

### 3. Launch the Search Engine UI (Streamlit)
```bash
cd frontend/
pip install -r requirements.txt
python -m streamlit run app.py
```
The UI talks to `http://localhost:8000/search` by default; set `BACKEND_URL` to point it elsewhere.

## 🛠️ Technology Stack
* **Vector DB Engine:** FAISS (`IndexFlatL2`)
* **Embedding Model:** 🤗 `sentence-transformers/all-MiniLM-L6-v2`
* **API Framework:** FastAPI, Uvicorn & Pydantic
* **Frontend:** Streamlit with Custom CSS injection
* **RAG LLM Engine:** Groq API (Llama-3 Architecture)
