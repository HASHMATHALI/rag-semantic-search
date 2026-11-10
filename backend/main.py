from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
        
        rag_generator = RAGGenerator()  # Uses GROQ_API_KEY from the environment if present
        indexer = SemanticIndexer(models_dir=models_dir)
    except Exception as e:
        init_error = str(e)
        print(f"FAILED to initialize services: {init_error}")

# Input request schema validation
class SearchQuery(BaseModel):
    query: str = Field(..., max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
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
    if init_error or not indexer:
        return {"status": "error", "detail": init_error or "Index not loaded. Run build_index.py first."}
    return {"status": "ok", "vectors_in_memory": indexer.index.ntotal if indexer else 0}

@app.post("/search", response_model=SearchResponse)
def perform_search(request: SearchQuery):
    if not indexer:
        raise HTTPException(
            status_code=503,
            detail=f"Search index is not loaded ({init_error or 'unknown error'}). Run `python build_index.py` in backend/.",
        )
        
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query validation failed: Please provide a query.")

    # 1. Conduct Semantic Retrieval (<100ms Inference speed locally)
    matches = indexer.search(request.query, request.top_k)
    
    rag_text = None
    # 2. Optionally conduct Generation Stage
    if request.use_rag and rag_generator:
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
