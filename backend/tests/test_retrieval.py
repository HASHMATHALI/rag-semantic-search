import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexer import BM25, mmr, reciprocal_rank_fusion, tokenize
from rag import RAGGenerator

DOCS = [
    "The central bank raised interest rates to fight inflation.",
    "A striker scored twice as the football club won the cup final.",
    "Interest in football is rising among younger fans.",
]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The Bank's rates, in 2024!") == ["bank", "s", "rates", "2024"]


def test_bm25_ranks_exact_term_matches_first():
    bm25 = BM25(DOCS)
    assert bm25.search("inflation rates", 3)[0][0] == 0
    assert bm25.search("football cup", 3)[0][0] == 1
    assert bm25.search("zebra", 3) == []


def test_rrf_rewards_agreement_between_engines():
    fused = reciprocal_rank_fusion({"semantic": [1, 2, 3], "lexical": [3, 1, 9]})
    assert max(fused, key=fused.get) == 1          # ranked 1st and 2nd
    assert fused[3] > fused[2] > fused[9]


def test_mmr_pushes_near_duplicates_down():
    vecs = np.array([[1, 0], [0.999, 0.04], [0, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    relevance = {10: 1.0, 11: 0.99, 12: 0.6}
    assert mmr([10, 11, 12], relevance, vecs, 2, lam=1.0) == [10, 11]   # pure relevance
    assert mmr([10, 11, 12], relevance, vecs, 2, lam=0.3) == [10, 12]   # diversity wins


def test_cited_sources_ignores_out_of_range_and_duplicates():
    assert RAGGenerator.cited_sources("Rates rose [2]. Again [2], see [1] and [9].", 3) == [2, 1]
    assert RAGGenerator.cited_sources(None, 3) == []
