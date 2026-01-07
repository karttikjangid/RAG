# RAG (Retrieval-Augmented Generation) System

A Python-based RAG pipeline for question-answering over multiple data sources using embeddings and local LLM generation.

## Overview

This project implements a complete RAG system that:
1. **Ingests data from multiple sources** (text files, PDFs, YouTube videos)
2. **Chunks text** using sliding window approach
3. **Creates vector embeddings** using Sentence Transformers
4. **Retrieves relevant context** based on semantic similarity
5. **Generates answers** using Ollama LLM

## Project Structure

```
RAG/
├── Core Modules
│   ├── main.py                 # Main application with multi-source support
│   ├── data_ingestion.py       # Text file ingestion
│   ├── youtube_ingestion.py    # YouTube transcript extraction
│   ├── pdf_ingestion.py        # PDF document processing
│   ├── chunking.py             # Text splitting with overlap
│   ├── embedding.py            # Vector embeddings creation
│   ├── retrieval.py            # Semantic search
│   └── generation.py           # LLM answer generation
│
├── Data
│   └── data.txt                # Sample text data (badminton info)
│
├── Documentation
│   ├── docs/                   # Detailed documentation
│   │   ├── RAG_SYSTEM_DOCUMENTATION.md
│   │   ├── COMPLETE_INGESTION_GUIDE.md
│   │   ├── YOUTUBE_INGESTION_DOCS.md
│   │   └── PDF_INGESTION_DOCS.md
│   └── README.md               # This file
│
└── Tests
    └── tests/                  # Test scripts
        └── test_real_rag.py
```

## Setup

### Prerequisites
- Python 3.8+
- Conda environment (recommended)
- Ollama installed locally

### Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install sentence-transformers
pip install scikit-learn
pip install requests
pip install youtube-transcript-api
pip install pypdf
```

3. **Install Ollama (optional for generation):**
- Download from [ollama.ai](https://ollama.ai)
- Pull a model:
```bash
ollama pull llama3.2
```

## Quick Start

Run the interactive RAG system:

```bash
python main.py
```

You'll be prompted to select a data source:
1. **Text File** - Load from .txt files
2. **PDF Document** - Extract text from PDFs
3. **YouTube Video** - Get transcripts from videos

Then ask questions about your data!

## Usage Examples

### Option 1: Interactive Mode (Recommended)

```bash
python main.py
```

Example session:
```
📚 SELECT DATA SOURCE:
   1. Text File (.txt)
   2. PDF Document (.pdf)
   3. YouTube Video (URL)

Enter your choice (1-3): 2
Enter PDF file path: ../my_document.pdf

✅ Successfully loaded 61,447 characters
✅ Created 410 chunks
✅ Generated 410 vectors

❓ Your question: What is RAG?
🔍 Searching knowledge base...
💬 Answer: RAG (Retrieval-Augmented Generation) is...
```

### Option 2: Programmatic Usage

#### Text File Ingestion
```python
from data_ingestion import reading_data
text = reading_data("data.txt")
```

#### PDF Document Ingestion
```python
from pdf_ingestion import get_pdf_text
text = get_pdf_text("document.pdf")
```

#### YouTube Transcript Ingestion
```python
from youtube_ingestion import get_youtube_transcript
text = get_youtube_transcript("https://www.youtube.com/watch?v=...")
```

#### Complete RAG Pipeline
```python
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import generate_answer

# Chunk the text
chunks = get_chunks(text, chunk_size=200, overlap=50)
# Chunk the text
chunks = get_chunks(text, chunk_size=200, overlap=50)

# Create embeddings
vectors, model = vector_embedding(chunks)

# Query the system
query = "What is RAG?"
results = search_best_chunks(query, model, vectors, chunks, k=3)

# Generate answer (requires Ollama)
answer = generate_answer(query, results[0]['text'])
print(answer)
```

## Features

- ✅ **Multi-source ingestion**: Text files, PDFs, YouTube videos
- ✅ **Sliding window chunking** with overlap
- ✅ **Vector embeddings** using Sentence Transformers (384-dim)
- ✅ **Semantic search** with cosine similarity
- ✅ **Local LLM generation** with Ollama
- ✅ **Interactive Q&A** interface
- ✅ **No API costs** - runs entirely offline

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

- **[RAG System Documentation](docs/RAG_SYSTEM_DOCUMENTATION.md)** - Complete system overview
- **[Ingestion Guide](docs/COMPLETE_INGESTION_GUIDE.md)** - Multi-source data ingestion
- **[YouTube Module](docs/YOUTUBE_INGESTION_DOCS.md)** - YouTube transcript extraction
- **[PDF Module](docs/PDF_INGESTION_DOCS.md)** - PDF processing details

## Configuration

### Data Source Options
1. **Text Files**: Any .txt file
2. **PDF Documents**: Multi-page PDF extraction
3. **YouTube Videos**: Automatic transcript fetching

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
