from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

from indexer import SemanticIndexer
from rag import RAGGenerator

# Initialize the Server Framework
app = FastAPI(title="Semantic RAG Search API", description="FastAPI Server hosting FAISS locally")

# Global variables so that indexing and model weights stay in RAM between queries 
indexer = None
rag_generator = None
init_error = None

# Using an startup event prevents holding up local variable space instantly
@app.on_event("startup")
def load_models():
    global indexer, rag_generator, init_error
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        
        indexer = SemanticIndexer(models_dir=models_dir)
        rag_generator = RAGGenerator() # Loads OPENAI_API_KEY from environment if present
    except Exception as e:
        init_error = str(e)
        print(f"FAILED to initialize services: {init_error}")

# Input request schema validation
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    use_rag: bool = False # Allows Streamlit UI to dynamically toggle generation!

# Output response schema validation
class SearchResult(BaseModel):
    id: int
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    rag_response: Optional[str] = None

@app.get("/health")
def health_check():
    """ Keeps an endpoint for deployment services like Render to ping for health monitors """
    if init_error:
        return {"status": "error", "detail": init_error}
    return {"status": "ok", "vectors_in_memory": indexer.index.ntotal if indexer else 0}

@app.post("/search", response_model=SearchResponse)
def perform_search(request: SearchQuery):
    if not indexer:
        raise HTTPException(status_code=500, detail="Search index failed to load globally.")
        
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query validation failed: Please provide a query.")

    # 1. Conduct Semantic Retrieval (<100ms Inference speed locally)
    matches = indexer.search(request.query, request.top_k)
    
    rag_text = None
    # 2. Optionally conduct Generation Stage
    if request.use_rag:
        contexts = [m['text'] for m in matches]
        rag_text = rag_generator.generate_response(request.query, contexts)
        
    return SearchResponse(
        query=request.query,
        results=matches,
        rag_response=rag_text
    )

if __name__ == "__main__":
    import uvicorn
    # Running uvicorn locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
