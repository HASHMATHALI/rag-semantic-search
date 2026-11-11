import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from indexer import SemanticIndexer
from rag import RAGGenerator

app = FastAPI(title="Cartograph API", description="Hybrid semantic search that knows the edge of its own map.")

# Loaded once at startup so model weights and the index stay in RAM between queries.
indexer = None
rag_generator = None
init_error = None


@app.on_event("startup")
def load_models():
    global indexer, rag_generator, init_error
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rag_generator = RAGGenerator()  # uses GROQ_API_KEY from the environment if present
        indexer = SemanticIndexer(models_dir=os.environ.get("MODELS_DIR", os.path.join(base_dir, "models")))
    except Exception as e:
        init_error = str(e)
        print(f"FAILED to initialize services: {init_error}")


class SearchQuery(BaseModel):
    query: str = Field(..., max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
    mode: Literal["hybrid", "semantic", "lexical"] = "hybrid"
    diversity: float = Field(0.0, ge=0.0, le=1.0, description="0 = pure relevance, 1 = maximally varied results")
    use_rag: bool = False


class SearchResult(BaseModel):
    id: int
    score: float          # squared L2 distance to the query; closer to 0 is better
    fused: float          # reciprocal-rank-fusion score used for ordering
    found_by: List[str]   # which engines returned this passage
    text: str
    evidence: str         # the sentence in the passage closest to the query
    label: int
    x: float
    y: float


class Coverage(BaseModel):
    best_distance: float
    best_cosine: float
    typical_distance: float
    score: float
    verdict: Literal["charted", "frontier", "uncharted"]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    coverage: Coverage
    query_xy: List[float]
    rag_response: Optional[str] = None
    cited: List[int] = []
    rag_skipped_reason: Optional[str] = None


def _require_index():
    if not indexer:
        raise HTTPException(
            status_code=503,
            detail=f"Search index is not loaded ({init_error or 'unknown error'}). Run `python build_index.py` in backend/.",
        )


@app.get("/health")
def health_check():
    """Liveness endpoint for hosts such as Render."""
    if init_error or not indexer:
        return {"status": "error", "detail": init_error or "Index not loaded. Run build_index.py first."}
    return {"status": "ok", "vectors_in_memory": indexer.index.ntotal, "answers_enabled": bool(rag_generator and rag_generator.enabled)}


@app.get("/map")
def corpus_map(limit: int = 1500):
    """A sample of the corpus projected to 2-D, for drawing the map."""
    _require_index()
    return {"points": indexer.map_points(max(100, min(limit, 5000))), "total": indexer.index.ntotal}


@app.post("/search", response_model=SearchResponse)
def perform_search(request: SearchQuery):
    _require_index()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query validation failed: Please provide a query.")

    found = indexer.search(request.query, request.top_k, request.mode, request.diversity)

    rag_text, cited, skipped = None, [], None
    if request.use_rag and rag_generator:
        if found["coverage"]["verdict"] == "uncharted":
            # Refusing here is the point: an LLM handed unrelated passages will still write something.
            skipped = "The query is outside what the indexed documents cover, so no answer was generated."
        else:
            contexts = [r["text"] for r in found["results"]]
            rag_text = rag_generator.generate_response(request.query, contexts)
            cited = rag_generator.cited_sources(rag_text, len(contexts))

    return SearchResponse(query=request.query, rag_response=rag_text, cited=cited, rag_skipped_reason=skipped, **found)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
