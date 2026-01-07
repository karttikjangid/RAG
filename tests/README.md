# Tests Directory

This directory contains test scripts for the RAG system.

## Available Tests

### test_real_rag.py
Interactive test using the real RAG PDF document.
- Tests PDF ingestion with 57-page document
- Creates embeddings and allows Q&A
- Demonstrates complete RAG pipeline

## Running Tests

```bash
# Activate virtual environment
source ../venv/bin/activate

# Run the RAG test
python test_real_rag.py
```

## Test Data

Tests use the RAG documentation PDF located in the parent directory:
- `../../Retrieval Augmented Generation (RAG) for Everyone (1).pdf`
