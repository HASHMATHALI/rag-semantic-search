"""Cartograph retrieval engine.

Hybrid retrieval (dense vectors + BM25, fused with reciprocal rank fusion),
MMR diversification, sentence-level evidence, a 2-D map of the corpus and a
data-derived "is this query even on the map?" coverage check.
"""
from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter, defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

TOKEN = re.compile(r"[a-z0-9]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over an in-memory inverted index (no extra dependencies)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.zeros(len(docs), dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            self.doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append((i, tf))
        self.avg_len = float(self.doc_len.mean()) if len(docs) else 0.0
        n = len(docs)
        self.index = {
            term: (
                np.fromiter((d for d, _ in plist), dtype=np.int32),
                np.fromiter((tf for _, tf in plist), dtype=np.float32),
                math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5)),
            )
            for term, plist in postings.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.index:
                continue
            ids, tf, idf = self.index[term]
            norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[ids] / (self.avg_len or 1.0))
            for doc_id, s in zip(ids, idf * tf * (self.k1 + 1) / norm):
                scores[int(doc_id)] += float(s)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    """Score = sum over engines of 1 / (k + rank). Needs no score normalisation."""
    fused: dict[int, float] = defaultdict(float)
    for ranked in rankings.values():
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return dict(fused)


def mmr(candidates: list[int], relevance: dict[int, float], vectors: np.ndarray, top_k: int, lam: float) -> list[int]:
    """Maximal marginal relevance: trade relevance against similarity to picks so far.

    ``vectors`` rows must be L2-normalised and aligned with ``candidates``.
    lam = 1 keeps pure relevance order; lower values push near-duplicates down.
    """
    if not candidates:
        return []
    rel = np.array([relevance[c] for c in candidates], dtype=np.float32)
    rel = rel / (rel.max() or 1.0)
    sim = vectors @ vectors.T
    chosen: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(chosen) < top_k:
        if not chosen:
            best = max(remaining, key=lambda i: rel[i])
        else:
            best = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * sim[i, chosen].max())
        chosen.append(best)
        remaining.remove(best)
    return [candidates[i] for i in chosen]


class SemanticIndexer:
    MIN_COSINE = 0.20   # below this the best hit shares essentially no meaning with the query
    WEAK_COSINE = 0.35

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", models_dir: str = "../models"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(models_dir, "faiss.index"))
        with open(os.path.join(models_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        self.texts = [m.get("text", "") for m in self.metadata]

        emb_path = os.path.join(models_dir, "embeddings.npy")
        self.embeddings = (
            np.load(emb_path).astype("float32") if os.path.exists(emb_path)
            else self.index.reconstruct_n(0, self.index.ntotal)
        )
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.unit = self.embeddings / np.where(norms == 0, 1, norms)

        self.bm25 = BM25(self.texts)
        self._fit_map()
        self._fit_coverage()

    # ------------------------------------------------------------------ map
    def _fit_map(self) -> None:
        """2-D PCA of the corpus: the 'map' queries are pinned on."""
        self.mean = self.embeddings.mean(axis=0)
        sample = self.embeddings[:: max(1, len(self.embeddings) // 5000)] - self.mean
        _, _, vt = np.linalg.svd(sample, full_matrices=False)
        self.axes = vt[:2]
        self.coords = (self.embeddings - self.mean) @ self.axes.T

    def project(self, vectors: np.ndarray) -> np.ndarray:
        return (vectors - self.mean) @ self.axes.T

    def map_points(self, limit: int = 1500) -> list[dict]:
        step = max(1, len(self.texts) // limit)
        return [
            {"id": i, "x": float(self.coords[i, 0]), "y": float(self.coords[i, 1]),
             "label": int(self.metadata[i].get("label", -1)), "text": self.texts[i][:90]}
            for i in range(0, len(self.texts), step)
        ]

    # ------------------------------------------------------------- coverage
    def _fit_coverage(self) -> None:
        """How far is each document from its nearest neighbour in the corpus?

        That distribution is the yardstick for queries: a query whose best hit is
        farther away than almost any document is from its neighbour is asking
        about something the corpus does not contain.
        """
        if self.index.ntotal < 2:
            self.neighbour_dist = np.array([0.0], dtype=np.float32)
            return
        dists, _ = self.index.search(self.embeddings, 2)
        self.neighbour_dist = np.sort(dists[:, 1])

    def coverage(self, best_distance: float, best_cosine: float) -> dict:
        """Two tests, and the stricter one wins.

        Relative: is the best hit farther away than documents usually are from
        their own nearest neighbour? Absolute: is its cosine similarity below
        MIN_COSINE? The absolute floor matters for small or sparse corpora, where
        documents are far from each other and the relative test alone is lenient.
        """
        n = len(self.neighbour_dist)
        # Share of documents that sit at least this far from their own nearest neighbour.
        isolation = 1.0 - float(np.searchsorted(self.neighbour_dist, best_distance, side="left")) / n
        if isolation < 0.02 or best_cosine < self.MIN_COSINE:
            verdict = "uncharted"      # nothing in the corpus is about this
        elif isolation < 0.25 or best_cosine < self.WEAK_COSINE:
            verdict = "frontier"       # only loosely related material exists
        else:
            verdict = "charted"        # as close to the corpus as documents are to each other
        return {"best_distance": float(best_distance), "best_cosine": round(float(best_cosine), 3),
                "score": round(isolation, 3), "verdict": verdict,
                "typical_distance": float(np.median(self.neighbour_dist))}

    # --------------------------------------------------------------- search
    def evidence(self, query_vec: np.ndarray, text: str) -> str:
        """The single sentence in ``text`` closest to the query."""
        sentences = [s.strip() for s in SENTENCE.split(text) if len(s.strip()) > 20]
        if len(sentences) < 2:
            return text.strip()
        vecs = self.model.encode(sentences, normalize_embeddings=True)
        q = query_vec / (np.linalg.norm(query_vec) or 1.0)
        return sentences[int(np.argmax(vecs @ q))]

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid", diversity: float = 0.0) -> dict:
        query_vec = np.asarray(self.model.encode([query]), dtype="float32")
        pool = min(max(top_k * 4, 20), self.index.ntotal)

        dists, ids = self.index.search(query_vec, pool)
        dense = [(int(i), float(d)) for d, i in zip(dists[0], ids[0]) if i != -1]
        dense_dist = dict(dense)
        lexical = self.bm25.search(query, pool) if mode in ("hybrid", "lexical") else []

        rankings = {}
        if mode in ("hybrid", "semantic"):
            rankings["semantic"] = [i for i, _ in dense]
        if mode in ("hybrid", "lexical"):
            rankings["lexical"] = [i for i, _ in lexical]
        fused = reciprocal_rank_fusion(rankings)
        candidates = sorted(fused, key=lambda i: -fused[i])[:pool]

        lam = 1.0 - max(0.0, min(1.0, diversity))
        order = mmr(candidates, fused, self.unit[candidates], top_k, lam) if candidates else []

        results = []
        for doc_id in order:
            found_by = [name for name, ranked in rankings.items() if doc_id in ranked[:pool]]
            dist = dense_dist.get(doc_id)
            if dist is None:  # lexical-only hit: compute its true distance for display
                dist = float(np.sum((self.embeddings[doc_id] - query_vec[0]) ** 2))
            results.append({
                "id": doc_id,
                "score": dist,
                "fused": round(fused[doc_id], 5),
                "found_by": found_by,
                "text": self.texts[doc_id],
                "evidence": self.evidence(query_vec[0], self.texts[doc_id]),
                "label": int(self.metadata[doc_id].get("label", -1)),
                "x": float(self.coords[doc_id, 0]),
                "y": float(self.coords[doc_id, 1]),
            })

        qxy = self.project(query_vec)[0]
        if dense:
            best_id, best = dense[0]
            best_cosine = float(self.unit[best_id] @ (query_vec[0] / (np.linalg.norm(query_vec[0]) or 1.0)))
        else:
            best, best_cosine = float("inf"), 0.0
        return {"results": results, "coverage": self.coverage(best, best_cosine),
                "query_xy": [float(qxy[0]), float(qxy[1])]}
