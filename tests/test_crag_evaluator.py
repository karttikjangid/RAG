from crag.config import CRAGConfig
from crag.retrieval_evaluator import evaluate_retrieval
from crag.types import ChunkScore


def test_retrieval_evaluator_relevant():
    config = CRAGConfig.from_env()
    chunks = [
        ChunkScore(
            index=0,
            text="alpha beta",
            vector_score=0.9,
            bm25_score=0.1,
            combined_score=0.9,
        )
    ]
    evaluation = evaluate_retrieval(chunks, config)
    assert evaluation.decision == "relevant"
