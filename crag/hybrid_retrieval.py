from typing import List, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import util

from .config import CRAGConfig
from .text_utils import keyword_tokens, tokenize
from .types import ChunkScore


def _bm25_tokens(text: str) -> List[str]:
    tokens = keyword_tokens(text)
    return tokens if tokens else tokenize(text)


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val < 1e-8:
        return [1.0 for _ in values]
    return [(val - min_val) / (max_val - min_val) for val in values]


class BM25Index:
    def __init__(self, tokenized_corpus: List[List[str]], bm25: BM25Okapi) -> None:
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25

    @classmethod
    def from_chunks(cls, chunks: List[str]) -> "BM25Index":
        tokenized = [_bm25_tokens(chunk) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        return cls(tokenized, bm25)

    def get_scores(self, query: str) -> np.ndarray:
        return self._bm25.get_scores(_bm25_tokens(query))

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        if len(scores) == 0:
            return []
        top_k = min(top_k, len(scores))
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices]


class HybridRetriever:
    def __init__(
        self,
        chunks: List[str],
        vectors,
        model,
        config: CRAGConfig,
        bm25_index: BM25Index = None,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors
        self._model = model
        self._config = config
        self._bm25_index = bm25_index or BM25Index.from_chunks(chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[ChunkScore]:
        top_k = top_k or self._config.top_k
        if not self._chunks:
            return []

        query_vector = self._model.encode(query)
        cosine_scores = util.cos_sim(query_vector, self._vectors)
        vector_scores = cosine_scores[0].tolist()

        vector_top_k = min(top_k, len(vector_scores))
        vector_top = torch.topk(torch.tensor(vector_scores), k=vector_top_k)
        vector_indices = [int(idx) for idx in vector_top.indices.tolist()]

        bm25_scores = np.zeros(len(self._chunks))
        bm25_indices: List[int] = []
        if self._config.enable_hybrid:
            bm25_scores = self._bm25_index.get_scores(query)
            bm25_top_k = min(top_k, len(bm25_scores))
            bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k].tolist()

        candidate_indices = sorted(set(vector_indices) | set(bm25_indices))

        vector_candidate_scores = [vector_scores[i] for i in candidate_indices]
        bm25_candidate_scores = [float(bm25_scores[i]) for i in candidate_indices]

        vector_norm = _normalize(vector_candidate_scores)
        bm25_norm = _normalize(bm25_candidate_scores)

        chunk_scores: List[ChunkScore] = []
        for idx, v_norm, b_norm in zip(
            candidate_indices, vector_norm, bm25_norm
        ):
            vector_score = float(vector_scores[idx])
            bm25_score = float(bm25_scores[idx])
            if self._config.enable_hybrid:
                combined = (
                    self._config.vector_weight * v_norm
                    + self._config.bm25_weight * b_norm
                )
                source = "hybrid"
            else:
                combined = v_norm
                source = "vector"
            chunk_scores.append(
                ChunkScore(
                    index=idx,
                    text=self._chunks[idx],
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    combined_score=combined,
                    metadata={"source": source},
                )
            )

        chunk_scores.sort(key=lambda item: item.combined_score, reverse=True)
        return chunk_scores
