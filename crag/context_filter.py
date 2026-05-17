from typing import List

from .config import CRAGConfig
from .text_utils import jaccard_similarity, token_set
from .types import ChunkScore


def filter_context(
    chunks: List[ChunkScore], config: CRAGConfig
) -> List[ChunkScore]:
    filtered: List[ChunkScore] = []
    for chunk in chunks:
        score = chunk.rerank_score or chunk.combined_score
        if score < config.min_context_score:
            continue

        chunk_tokens = token_set(chunk.text)
        is_duplicate = False
        for existing in filtered:
            existing_tokens = token_set(existing.text)
            if (
                jaccard_similarity(chunk_tokens, existing_tokens)
                >= config.dedupe_jaccard
            ):
                is_duplicate = True
                break
        if is_duplicate:
            continue

        filtered.append(chunk)
        if len(filtered) >= config.max_context_chunks:
            break

    return filtered
