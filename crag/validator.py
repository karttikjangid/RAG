from typing import List, Optional

from .config import CRAGConfig
from .text_utils import keyword_tokens, split_sentences
from .types import AnswerValidation


def validate_answer(
    answer: str,
    context: str,
    config: CRAGConfig,
    no_answer_response: Optional[str] = None,
) -> AnswerValidation:
    if not answer or not context:
        return AnswerValidation(
            is_grounded=False, grounding_score=0.0, unsupported_sentences=[]
        )

    if no_answer_response and answer.strip() == no_answer_response:
        return AnswerValidation(
            is_grounded=True, grounding_score=1.0, unsupported_sentences=[]
        )

    context_tokens = set(keyword_tokens(context))
    sentences = split_sentences(answer)

    if not sentences:
        return AnswerValidation(
            is_grounded=False, grounding_score=0.0, unsupported_sentences=[]
        )

    supported = 0
    unsupported: List[str] = []

    for sentence in sentences:
        sentence_tokens = set(keyword_tokens(sentence))
        if not sentence_tokens:
            continue
        overlap = len(sentence_tokens & context_tokens) / len(sentence_tokens)
        if overlap >= config.min_sentence_overlap:
            supported += 1
        else:
            unsupported.append(sentence)

    grounding_score = supported / len(sentences)
    is_grounded = grounding_score >= config.min_grounded_sentence_ratio

    return AnswerValidation(
        is_grounded=is_grounded,
        grounding_score=grounding_score,
        unsupported_sentences=unsupported,
    )
