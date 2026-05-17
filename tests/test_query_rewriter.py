from crag.config import CRAGConfig
from crag.query_rewriter import rewrite_query


def test_query_rewriter_expands_keywords():
    config = CRAGConfig.from_env()
    rewrites = rewrite_query("What is RAG?", config)
    assert rewrites[0] == "What is RAG?"
    assert any("rag" in rewrite.lower() for rewrite in rewrites)
