# RAG (Retrieval-Augmented Generation) System

A Python-based RAG pipeline for question-answering over text documents using embeddings and local LLM generation.

## Overview

This project implements a complete RAG system that:
1. Ingests text data
2. Chunks text using sliding window approach
3. Creates vector embeddings using Sentence Transformers
4. Retrieves relevant context based on queries
5. Generates answers using Ollama LLM

## Project Structure

```
RAG/
├── data_ingestion.py   # Read and load text data
├── chunking.py         # Split text into overlapping chunks
├── embedding.py        # Create vector embeddings
├── generation.py       # Generate answers using Ollama
├── data.txt           # Source text (badminton information)
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Setup

### Prerequisites
- Python 3.8+
- Conda environment (recommended)
- Ollama installed locally

### Installation

1. **Create and activate conda environment:**
```bash
conda create -n d2l python=3.10
conda activate rag_env
```

2. **Install dependencies:**
```bash
pip install sentence-transformers
pip install scikit-learn
pip install requests
```

3. **Install Ollama:**
- Download from [ollama.ai](https://ollama.ai)
- Pull a model:
```bash
ollama pull llama3.2
```

### Fix PyTorch/OpenMP Issues

If you encounter OpenMP errors, set the environment variable:

**Windows (cmd):**
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```

**Windows (PowerShell):**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

**Permanent fix (conda):**
```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n d2l
conda activate d2l
```

## Usage

### 1. Data Ingestion
```python
from data_ingestion import reading_data

text = reading_data("data.txt")
```

### 2. Chunking
```python
from chunking import get_chunks

chunks = get_chunks(text, chunk_size=200, overlap=50)
print(f"Created {len(chunks)} chunks")
```

### 3. Create Embeddings
```python
from embedding import vector_embedding

vectors, model = vector_embedding(chunks)
print(f"Shape: {vectors[0].shape}")
```

### 4. Generate Answers
```python
from generation import generate_answer

query = "When did badminton join the Olympics?"
context = "Badminton debuted in the Olympics in 1992."
answer = generate_answer(query, context)
print(answer)
```

## Configuration

### Chunking Parameters
- `chunk_size`: Characters per chunk (default: 200)
- `overlap`: Overlapping characters between chunks (default: 50)

### Embedding Model
- Default: `all-MiniLM-L6-v2` (384 dimensions)
- Can be changed in `embedding.py`

### LLM Model
- Default: `llama3.2`
- Can be changed to `mistral` or other Ollama models in `generation.py`

## Troubleshooting

### PyTorch Version Issues
- Requires PyTorch >= 2.1
- Check version: `python -c "import torch; print(torch.__version__)"`
- Upgrade: `pip install --upgrade torch`

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check it's on port 11434
- Test: `curl http://localhost:11434`

### Module Not Found
- Activate conda environment: `conda activate d2l`
- Reinstall packages: `pip install sentence-transformers`

## Features

- ✅ Sliding window text chunking with overlap
- ✅ Vector embeddings using Sentence Transformers
- ✅ Local LLM generation with Ollama
- ✅ Context-aware question answering
- ✅ No external API costs

## Future Enhancements

- [ ] Add vector database (FAISS/ChromaDB) for retrieval
- [ ] Implement similarity search for finding relevant chunks
- [ ] Add support for multiple document types (PDF, DOCX)
- [ ] Web interface using Streamlit/Gradio
- [ ] Evaluation metrics for answer quality

## License

MIT

## Author

Built for learning RAG systems and local LLM integration.
