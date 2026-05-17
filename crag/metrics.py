from typing import Iterable, List

from .text_utils import keyword_tokens
from .types import AnswerValidation, ChunkScore


def retrieval_precision(
    retrieved_chunks: List[ChunkScore], relevant_phrases: Iterable[str]
) -> float:
    relevant = [phrase.lower() for phrase in relevant_phrases]
    if not retrieved_chunks:
        return 0.0
    matches = 0
    for chunk in retrieved_chunks:
        text = chunk.text.lower()
        if any(phrase in text for phrase in relevant):
            matches += 1
    return matches / len(retrieved_chunks)


def answer_grounding_rate(validations: Iterable[AnswerValidation]) -> float:
    validations = list(validations)
    if not validations:
        return 0.0
    grounded = sum(1 for validation in validations if validation.is_grounded)
    return grounded / len(validations)


def response_relevance(answer: str, query: str) -> float:
    answer_tokens = set(keyword_tokens(answer))
    query_tokens = set(keyword_tokens(query))
    if not answer_tokens or not query_tokens:
        return 0.0
    return len(answer_tokens & query_tokens) / len(query_tokens)


def hallucination_rate(validations: Iterable[AnswerValidation]) -> float:
    validations = list(validations)
    if not validations:
        return 0.0
    hallucinated = sum(1 for validation in validations if not validation.is_grounded)
    return hallucinated / len(validations)
