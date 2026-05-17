from dataclasses import replace

import numpy as np

from crag.config import CRAGConfig
from crag.hybrid_retrieval import HybridRetriever


class DummyModel:
    def __init__(self, vector):
        self._vector = vector

    def encode(self, text):
        return self._vector


def test_hybrid_retriever_combines_scores():
    config = replace(
        CRAGConfig.from_env(),
        top_k=2,
        vector_weight=0.5,
        bm25_weight=0.5,
        enable_hybrid=True,
    )
    chunks = ["alpha beta", "gamma delta", "alpha gamma"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    model = DummyModel(np.array([1.0, 0.0]))

    retriever = HybridRetriever(chunks, vectors, model, config)
    results = retriever.retrieve("alpha", top_k=2)

    assert len(results) == 2
    assert results[0].index in {0, 2}
