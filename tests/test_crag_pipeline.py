from dataclasses import replace

import numpy as np

from generation import NO_ANSWER_RESPONSE
from crag.config import CRAGConfig
from crag.controller import CorrectiveRAGController
from crag.hybrid_retrieval import HybridRetriever


class DummyModel:
    def __init__(self, vector):
        self._vector = vector

    def encode(self, text):
        return self._vector


def fake_generator(query, context, strict=False):
    if "alpha" in context.lower():
        return "Alpha appears in the context."
    return NO_ANSWER_RESPONSE


def test_crag_pipeline_returns_answer():
    config = replace(
        CRAGConfig.from_env(),
        top_k=2,
        max_retries=1,
        enable_validation=True,
    )
    chunks = ["alpha beta", "gamma delta"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = DummyModel(np.array([1.0, 0.0]))

    retriever = HybridRetriever(chunks, vectors, model, config)
    controller = CorrectiveRAGController(
        config, retriever, generator=fake_generator
    )

    response = controller.run("alpha")
    assert "Alpha" in response.answer


def test_crag_pipeline_returns_fallback_on_empty_retrieval():
    class EmptyRetriever:
        def retrieve(self, query, top_k=None):
            return []

    config = replace(CRAGConfig.from_env(), max_retries=0)
    controller = CorrectiveRAGController(
        config, EmptyRetriever(), generator=fake_generator
    )

    response = controller.run("unknown")
    assert response.answer.startswith(NO_ANSWER_RESPONSE)
