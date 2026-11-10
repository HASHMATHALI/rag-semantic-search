"""Build the FAISS index locally (no Colab needed).

Usage:
    python build_index.py                      # small bundled sample corpus (seconds)
    python build_index.py --ag-news 25000      # AG News subset (needs `pip install datasets`)
    python build_index.py --file docs.txt      # your own corpus, one document per line

Writes faiss.index, embeddings.npy and metadata.pkl into ../models/.
"""
import argparse
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_FILE = os.path.join(BASE_DIR, "data", "sample_docs.txt")


def load_lines(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{"text": line.strip(), "label": -1} for line in f if line.strip()]


def load_ag_news(limit: int) -> list[dict]:
    from datasets import load_dataset  # optional dependency

    dataset = load_dataset("ag_news", split=f"train[:{limit}]")
    return [{"text": row["text"].strip(), "label": int(row["label"])} for row in dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="Text file with one document per line")
    source.add_argument("--ag-news", type=int, metavar="N", help="Index the first N AG News articles")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    if args.ag_news:
        records = load_ag_news(args.ag_news)
    else:
        records = load_lines(args.file or SAMPLE_FILE)
    if not records:
        raise SystemExit("No documents found to index.")

    print(f"Embedding {len(records)} documents with {args.model}...")
    model = SentenceTransformer(args.model)
    embeddings = np.asarray(
        model.encode([r["text"] for r in records], batch_size=256, show_progress_bar=len(records) > 500),
        dtype="float32",
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(MODELS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(MODELS_DIR, "faiss.index"))
    np.save(os.path.join(MODELS_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(MODELS_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(records, f)
    print(f"Done. {index.ntotal} vectors written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
