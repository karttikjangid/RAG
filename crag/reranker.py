from typing import List

from .config import CRAGConfig
from .text_utils import keyword_tokens
from .types import ChunkScore


def rerank_chunks(
    query: str, chunks: List[ChunkScore], config: CRAGConfig
) -> List[ChunkScore]:
    if not chunks:
        return []

    query_tokens = set(keyword_tokens(query))
    weight = min(max(config.rerank_lexical_weight, 0.0), 1.0)

    for chunk in chunks:
        if not query_tokens:
            overlap = 0.0
        else:
            chunk_tokens = set(keyword_tokens(chunk.text))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)

        chunk.rerank_score = (1 - weight) * chunk.combined_score + weight * overlap

    return sorted(chunks, key=lambda item: item.rerank_score or 0.0, reverse=True)
