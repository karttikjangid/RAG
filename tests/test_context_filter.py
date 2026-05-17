from dataclasses import replace

from crag.config import CRAGConfig
from crag.context_filter import filter_context
from crag.types import ChunkScore


def test_context_filter_deduplicates():
    config = replace(
        CRAGConfig.from_env(),
        min_context_score=0.1,
        max_context_chunks=3,
        dedupe_jaccard=0.8,
    )
    chunks = [
        ChunkScore(0, "alpha beta", 0.5, 0.1, 0.5),
        ChunkScore(1, "alpha beta", 0.4, 0.1, 0.4),
        ChunkScore(2, "gamma delta", 0.3, 0.1, 0.3),
    ]

    filtered = filter_context(chunks, config)
    assert len(filtered) == 2
