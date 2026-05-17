from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkScore:
    index: int
    text: str
    vector_score: float
    bm25_score: float
    combined_score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    rewritten_query: Optional[str]
    attempt: int
    used_hybrid: bool
    chunks: List[ChunkScore]


@dataclass
class RetrievalEvaluation:
    decision: str
    max_score: float
    avg_score: float
    reason: str


@dataclass
class AnswerValidation:
    is_grounded: bool
    grounding_score: float
    unsupported_sentences: List[str]


@dataclass
class CRAGResponse:
    answer: str
    context: str
    evaluation: RetrievalEvaluation
    retrieval: RetrievalResult
    validation: Optional[AnswerValidation]
    attempts: List[Dict[str, Any]]
