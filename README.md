# Cartograph — search that knows where its map ends

Semantic search always returns *something*. Ask a recipe question of a news archive and you still
get five confident-looking passages, and a RAG pipeline will happily write an answer from them.

Cartograph treats that as the main problem. Every query is placed on a map of the indexed corpus
and given a bearing — **Charted waters**, **Frontier** or **Off the map** — before any passage or
generated answer is shown. Off-the-map queries are never sent to the LLM.

## What makes it different

- **Coverage verdict from the corpus itself.** At startup the engine measures how far every
  passage is from its own nearest neighbour. A query whose best hit is farther away than ~98% of
  passages are from theirs is off the map. An absolute cosine floor (0.20) backs this up for small
  corpora. No hand-tuned score threshold per dataset.
- **A map you can read.** All passages are projected to two dimensions; your query is the coral
  cross, retrieved passages are numbered, and lines show what was pulled from where.
- **Two engines, shown separately.** Dense vectors (FAISS, `all-MiniLM-L6-v2`) and BM25 are fused
  with reciprocal rank fusion. Each result says whether meaning, exact words or both found it.
- **Evidence sentence.** In multi-sentence passages the sentence closest to the query is highlighted.
- **Spread control.** Maximal marginal relevance pushes near-duplicates down when you want to see
  more of the neighbourhood.
- **Answers that cite or stay silent.** With `GROQ_API_KEY` set, answers must cite passages as
  `[n]`; cited passages are tagged in the list.

## Run it

```bash
# 1. build an index (not stored in git)
cd backend
pip install -r requirements.txt
python build_index.py                  # bundled 36-passage sample corpus
python build_index.py --ag-news 25000  # 25k news articles (needs `pip install datasets`)
python build_index.py --file docs.txt  # your own corpus, one passage per line

# 2. API
python -m uvicorn main:app --port 8000       # optional: set GROQ_API_KEY first for answers

# 3. UI (second terminal)
cd ../frontend
pip install -r requirements.txt
streamlit run app.py                          # BACKEND_URL overrides http://localhost:8000
```

```bash
cd backend && pip install pytest httpx && pytest -q   # 9 tests, including end-to-end on the sample corpus
```

## API

| Endpoint | Purpose |
| :-- | :-- |
| `POST /search` | `{query, top_k, mode: hybrid\|semantic\|lexical, diversity: 0-1, use_rag}` → results with `found_by`, `evidence`, map coordinates, plus `coverage` and optional cited answer |
| `GET /map` | Sampled 2-D projection of the corpus |
| `GET /health` | Vector count and whether answer generation is enabled |

## Limits

The 2-D map keeps only the two strongest directions of a 384-dimensional space, so it is a guide,
not a measurement. BM25 and the nearest-neighbour baseline are built in memory at startup, which is
comfortable to roughly 50k passages. The coverage verdict detects "nothing here is about this"; it
does not check whether a passage is *correct*.

For very large corpora, `colab/` contains the original notebook-style pipeline for building the
index on a free Colab instance.
