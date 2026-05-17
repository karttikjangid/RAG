from crag.config import CRAGConfig
from crag.validator import validate_answer


def test_validator_detects_grounded_answer():
    config = CRAGConfig.from_env()
    context = "RAG stands for retrieval augmented generation."
    answer = "RAG stands for retrieval augmented generation."
    validation = validate_answer(answer, context, config)
    assert validation.is_grounded is True


def test_validator_flags_unsupported_answer():
    config = CRAGConfig.from_env()
    context = "RAG stands for retrieval augmented generation."
    answer = "RAG was invented in 1901."
    validation = validate_answer(answer, context, config)
    assert validation.is_grounded is False
