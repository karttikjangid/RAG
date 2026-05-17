from typing import List

from .config import CRAGConfig
from .types import ChunkScore, RetrievalEvaluation


def evaluate_retrieval(
    chunks: List[ChunkScore], config: CRAGConfig
) -> RetrievalEvaluation:
    if not chunks:
        return RetrievalEvaluation(
            decision="irrelevant",
            max_score=0.0,
            avg_score=0.0,
            reason="no_chunks",
        )

    vector_scores = [chunk.vector_score for chunk in chunks]
    max_score = max(vector_scores)
    top_count = min(3, len(vector_scores))
    avg_score = sum(vector_scores[:top_count]) / top_count

    if max_score >= config.relevant_threshold:
        decision = "relevant"
    elif max_score >= config.partial_threshold:
        decision = "partially_relevant"
    else:
        decision = "irrelevant"

    reason = f"max_vector={max_score:.3f}, avg_vector={avg_score:.3f}"
    return RetrievalEvaluation(
        decision=decision, max_score=max_score, avg_score=avg_score, reason=reason
    )
