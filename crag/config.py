import os
from dataclasses import dataclass
from typing import Optional


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CRAGConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chunks: int
    max_retries: int
    relevant_threshold: float
    partial_threshold: float
    min_context_score: float
    vector_weight: float
    bm25_weight: float
    rerank_lexical_weight: float
    dedupe_jaccard: float
    min_grounded_sentence_ratio: float
    min_sentence_overlap: float
    max_generation_attempts: int
    enable_query_rewrite: bool
    enable_hybrid: bool
    enable_validation: bool
    log_level: str
    min_query_tokens: int

    @classmethod
    def from_env(cls) -> "CRAGConfig":
        vector_weight = _get_env_float("CRAG_VECTOR_WEIGHT", 0.7)
        bm25_weight: Optional[float] = None
        raw_bm25_weight = os.getenv("CRAG_BM25_WEIGHT")
        if raw_bm25_weight is not None:
            try:
                bm25_weight = float(raw_bm25_weight)
            except ValueError:
                bm25_weight = None
        if bm25_weight is None:
            bm25_weight = max(0.0, 1.0 - vector_weight)

        total_weight = vector_weight + bm25_weight
        if total_weight <= 0:
            vector_weight = 0.7
            bm25_weight = 0.3
        else:
            vector_weight = vector_weight / total_weight
            bm25_weight = bm25_weight / total_weight

        return cls(
            chunk_size=_get_env_int("CHUNK_SIZE", 600),
            chunk_overlap=_get_env_int("CHUNK_OVERLAP", 120),
            top_k=_get_env_int("CRAG_TOP_K", 3),
            max_context_chunks=_get_env_int("CRAG_MAX_CONTEXT_CHUNKS", 4),
            max_retries=_get_env_int("CRAG_MAX_RETRIES", 2),
            relevant_threshold=_get_env_float("CRAG_RELEVANT_THRESHOLD", 0.35),
            partial_threshold=_get_env_float("CRAG_PARTIAL_THRESHOLD", 0.2),
            min_context_score=_get_env_float("CRAG_MIN_CONTEXT_SCORE", 0.15),
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            rerank_lexical_weight=_get_env_float("CRAG_RERANK_LEXICAL_WEIGHT", 0.2),
            dedupe_jaccard=_get_env_float("CRAG_DEDUPE_JACCARD", 0.9),
            min_grounded_sentence_ratio=_get_env_float(
                "CRAG_MIN_GROUNDED_SENTENCE_RATIO", 0.6
            ),
            min_sentence_overlap=_get_env_float("CRAG_MIN_SENTENCE_OVERLAP", 0.2),
            max_generation_attempts=_get_env_int("CRAG_MAX_GENERATION_ATTEMPTS", 2),
            enable_query_rewrite=_get_env_bool("CRAG_ENABLE_QUERY_REWRITE", True),
            enable_hybrid=_get_env_bool("CRAG_ENABLE_HYBRID", True),
            enable_validation=_get_env_bool("CRAG_ENABLE_VALIDATION", True),
            log_level=os.getenv("CRAG_LOG_LEVEL", "INFO"),
            min_query_tokens=_get_env_int("CRAG_MIN_QUERY_TOKENS", 3),
        )
