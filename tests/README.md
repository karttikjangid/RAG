# Tests Directory

This directory contains automated and manual test scripts for the RAG system.

## Available Tests

### test_real_rag.py
Interactive test using the real RAG PDF document.
- Tests PDF ingestion with the 57-page document
- Creates embeddings and allows Q&A
- Demonstrates the Corrective RAG pipeline

### Automated CRAG Tests
- `test_crag_evaluator.py` - retrieval quality classification
- `test_query_rewriter.py` - query rewrite coverage
- `test_context_filter.py` - deduping and confidence thresholds
- `test_validator.py` - answer grounding detection
- `test_hybrid_retriever.py` - BM25 + vector score fusion
- `test_crag_pipeline.py` - end-to-end controller behavior

## Running Tests

```bash
# Activate virtual environment
source ../venv/bin/activate

# Run automated tests
pytest

# Run the manual demo
python test_real_rag.py
```

## Test Data

Manual tests use the RAG documentation PDF located in the parent directory:
- `../../Retrieval Augmented Generation (RAG) for Everyone (1).pdf`
