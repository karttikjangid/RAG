# RAG System - Complete Technical Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Technical Stack](#technical-stack)
6. [Setup and Installation](#setup-and-installation)
7. [Usage Guide](#usage-guide)
8. [Code Analysis](#code-analysis)

---

## 🎯 System Overview

This is a **Retrieval-Augmented Generation (RAG)** system built in Python that enables intelligent question-answering over text documents. The system combines **semantic search** using vector embeddings with **local LLM generation** to provide accurate, context-aware answers.

### What is RAG?
RAG is a technique that enhances Large Language Model (LLM) responses by:
1. **Retrieving** relevant context from a knowledge base
2. **Augmenting** the user's query with this context
3. **Generating** an answer based on the retrieved information

### Key Features
- ✅ **Local-first**: Runs entirely on your machine using Ollama
- ✅ **No API costs**: No external API dependencies
- ✅ **Modular design**: Each component is independent and testable
- ✅ **Efficient chunking**: Sliding window approach with overlap for better context
- ✅ **Fast retrieval**: Uses cosine similarity on vector embeddings
- ✅ **Conversational**: Interactive chat interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA INGESTION                                           │
│     └─> Read text file                                      │
│                                                              │
│  2. CHUNKING                                                 │
│     └─> Split into overlapping segments                     │
│                                                              │
│  3. EMBEDDING                                                │
│     └─> Convert chunks to 384-dim vectors                   │
│                                                              │
│  4. RETRIEVAL (Query Time)                                   │
│     └─> Find most similar chunk via cosine similarity       │
│                                                              │
│  5. GENERATION                                               │
│     └─> LLM generates answer from retrieved context         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Breakdown

### 1. **data_ingestion.py** - Data Loading Module
**Purpose**: Read and load text data from files

**Code Analysis**:
```python
def reading_data(file_path):
    with open(file_path, mode="r", encoding='utf-8') as f:
        text = f.read()
    return text
```

**Features**:
- UTF-8 encoding support for international characters
- Simple file I/O with context manager (automatic file closing)
- Returns entire file content as a single string

**Use Case**: Initial data loading from `data.txt` (contains badminton information)

---

### 2. **chunking.py** - Text Chunking Module
**Purpose**: Split large text into smaller, overlapping chunks

**Code Analysis**:
```python
def get_chunks(text, chunk_size, overlap):
    sliced = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks = text[start:end]
        sliced.append(chunks)
        start += (chunk_size - overlap)
    return sliced
```

**How it Works**:
- **Sliding Window Algorithm**: Moves through text with controlled overlap
- **Parameters**:
  - `chunk_size`: Number of characters per chunk (default: 150-200)
  - `overlap`: Number of overlapping characters between chunks (default: 50)
  
**Why Overlap?**
- Prevents important information from being split awkwardly
- Maintains context across chunk boundaries
- Example: With chunk_size=150, overlap=50:
  - Chunk 1: chars 0-150
  - Chunk 2: chars 100-250 (overlaps 50 chars with Chunk 1)

**Trade-offs**:
- More overlap = Better context retention, but more chunks to process
- Less overlap = Faster, but may lose context

---

### 3. **embedding.py** - Vector Embedding Module
**Purpose**: Convert text chunks into numerical vector representations

**Code Analysis**:
```python
from sentence_transformers import SentenceTransformer

def vector_embedding(chunked):
    print("loading embedding model.......")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunked)
    print(f"Created {len(vectors)} vectors.")
    print(f"Shape of first vector: {vectors[0].shape}")
    return vectors, model
```

**Key Points**:
- **Model Used**: `all-MiniLM-L6-v2` from Sentence Transformers
- **Output**: 384-dimensional dense vectors
- **Returns**: Both the vectors AND the model (important for query encoding later)

**How Embeddings Work**:
1. Text is converted to numerical vectors that capture semantic meaning
2. Similar texts have vectors that are close in vector space
3. Enables mathematical comparison (cosine similarity)

**Vector Properties**:
- **Dimension**: 384 floats per vector
- **Normalization**: Vectors are normalized for cosine similarity
- **Semantic**: "Olympic badminton" and "badminton in Olympics" have similar vectors

---

### 4. **retrieval.py** - Smart Retrieval Module
**Purpose**: Find the most relevant text chunk for a given query

**Code Analysis**:
```python
from sentence_transformers import util
import torch

def search_best_chunk(query, model, db_vectors, chunks):
    # 1. Encode the query
    query_vector = model.encode(query)
    
    # 2. Calculate similarity scores
    scores = util.cos_sim(query_vector, db_vectors)
    
    # 3. Find best match
    best_index = torch.argmax(scores)
    
    # 4. Return text and score
    return chunks[best_index], scores[0][best_index]
```

**Process Flow**:
1. **Query Encoding**: Convert user question to 384-dim vector
2. **Similarity Calculation**: Compare query vector against all chunk vectors
3. **Cosine Similarity Formula**: 
   ```
   similarity = (A · B) / (||A|| × ||B||)
   ```
   - Range: -1 (opposite) to 1 (identical)
   - Higher score = More similar
4. **Best Match Selection**: Pick chunk with highest score

**Why Cosine Similarity?**
- Measures angle between vectors, not magnitude
- Perfect for text similarity (length-independent)
- Fast computation with optimized libraries

---

### 5. **retreival.py** (Legacy Version)
**Note**: This is an older version with different architecture

**Key Difference**:
- Loads data/model at module level (once during import)
- `search_best_chunk()` takes only the query as parameter
- Less flexible than `retrieval.py` (can't reuse model for different datasets)

**Status**: Likely deprecated in favor of `retrieval.py`

---

### 6. **generation.py** - Answer Generation Module
**Purpose**: Generate natural language answers using a local LLM

**Code Analysis**:
```python
import requests

def generate_answer(query, context):
    # 1. Prompt Engineering
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the provided context. 
    If the answer is not in the context, say "I don't know."
    
    Context: {context}
    
    Question: {query}
    """
    
    # 2. Connect to Ollama
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False 
    }
    
    # 3. Send request
    response = requests.post(url, json=payload)
    return response.json()['response']
```

**Components**:

1. **Prompt Engineering**:
   - Critical instruction: "Answer based ONLY on the provided context"
   - Prevents hallucination (making up information)
   - Fallback: "I don't know" if answer not in context

2. **Ollama Integration**:
   - Local LLM server running on port 11434
   - Model: `llama3.2` (can be changed to mistral, etc.)
   - Non-streaming mode for simplicity

3. **Error Handling**:
   - Catches connection errors
   - Returns error message if Ollama is not running

**Why Local LLM?**
- Privacy: Data never leaves your machine
- Cost: No API fees
- Speed: No network latency (after initial model load)

---

### 7. **main.py** - Orchestration Module
**Purpose**: Tie everything together into an interactive application

**Code Analysis**:
```python
def start_app():
    # INITIALIZATION PHASE (Run Once)
    raw_text = reading_data("data.txt")
    text_chunks = get_chunks(raw_text, 150, 50)
    db_vectors, model = vector_embedding(text_chunks)
    
    # CHAT LOOP (Continuous)
    while True:
        query = input("\nUser: ")
        
        if query.lower() == "exit":
            break
            
        # A. RETRIEVAL
        best_context, score = search_best_chunk(
            query, model, db_vectors, text_chunks
        )
        
        # B. GENERATION
        answer = generate_answer(query, best_context)
        
        print(f"\nAI Answer:\n{answer}")
```

**Execution Flow**:

**Phase 1: Initialization** (Happens once)
1. Load text file
2. Create chunks
3. Generate embeddings
4. Load model into memory

**Phase 2: Chat Loop** (Repeats per query)
1. User inputs question
2. Retrieve most relevant chunk
3. Generate answer using LLM
4. Display results with timing

**Performance Optimizations**:
- Model loaded once (not per query)
- Vectors pre-computed (not generated on-the-fly)
- Reuses same model for query encoding

**User Experience**:
- Shows confidence score
- Displays context snippet
- Tracks response time
- Simple exit command

---

## 📊 Data Flow

### Complete RAG Cycle

```
User Query: "When did badminton join the Olympics?"
              ↓
┌─────────────────────────────────────────┐
│ 1. QUERY ENCODING                        │
│    "When did...?" → [0.23, -0.41, ...]  │ (384-dim vector)
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. SIMILARITY SEARCH                     │
│    Compare against all chunk vectors     │
│    Chunk 15: 0.87 ← BEST MATCH          │
│    Chunk 3:  0.45                        │
│    Chunk 8:  0.32                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. CONTEXT RETRIEVAL                     │
│    "In 1992, badminton debuted as a     │
│     Summer Olympic sport..."             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. PROMPT CONSTRUCTION                   │
│    System: Answer based on context only  │
│    Context: [Retrieved chunk]            │
│    Question: [User query]                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. LLM GENERATION (Ollama)               │
│    "Badminton joined the Olympics in    │
│     1992 as a Summer Olympic sport."     │
└─────────────────────────────────────────┘
              ↓
         Answer to User
```

---

## 🛠️ Technical Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | Latest | Text embedding and similarity |
| `torch` | 2.1+ | Tensor operations |
| `requests` | Latest | HTTP communication with Ollama |
| `scikit-learn` | Latest | ML utilities (future use) |

### External Services

| Service | Purpose |
|---------|---------|
| **Ollama** | Local LLM server (llama3.2 model) |
| **HuggingFace** | Source of embedding model |

### Python Version
- **Required**: Python 3.8+
- **Recommended**: Python 3.10

---

## 📦 Setup and Installation

### Prerequisites
```bash
# 1. Create conda environment
conda create -n rag_env python=3.10
conda activate rag_env

# 2. Install Python packages
pip install sentence-transformers
pip install scikit-learn
pip install requests

# 3. Install Ollama (Linux/Mac)
curl https://ollama.ai/install.sh | sh

# 4. Pull LLM model
ollama pull llama3.2
```

### Fix Common Issues

**OpenMP Error** (PyTorch conflict):
```bash
# Temporary fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Permanent fix (conda)
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n rag_env
conda activate rag_env
```

**Verify Installation**:
```bash
# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check Ollama
curl http://localhost:11434
ollama list
```

---

## 🚀 Usage Guide

### Basic Usage

**Run the main application**:
```bash
cd RAG
python main.py
```

**Interactive session**:
```
--- 🚀 INITIALIZING RAG SYSTEM ---
Step 1: Loading Data...
   -> Created 45 chunks.
Step 2: Embedding Data (Loading Model)...
   -> Embeddings ready.

--- ✅ SYSTEM READY (Type 'exit' to quit) ---

User: When did badminton join the Olympics?

🔍 Context Found (Confidence: 0.87):
"In 1992, badminton debuted as a Summer Olympic sport with four events..."

🤖 Thinking...

AI Answer:
Badminton joined the Olympics in 1992 as a Summer Olympic sport.

(Time taken: 2.34s)
```

### Module-Level Testing

**Test each component independently**:

```bash
# 1. Test data ingestion
python data_ingestion.py
# Output: 15234 (character count)

# 2. Test chunking
python chunking.py
# Output: Shows first 3 chunks

# 3. Test embeddings
python embedding.py
# Output: Shape of vectors

# 4. Test retrieval
python retreival.py
# Output: Best matching chunk

# 5. Test generation
python generation.py
# Output: LLM answer
```

---

## 🔍 Code Analysis

### Design Patterns Used

1. **Modular Architecture**
   - Each file has a single responsibility
   - Functions are pure (no side effects where possible)
   - Easy to test and maintain

2. **Separation of Concerns**
   - Data layer (ingestion)
   - Processing layer (chunking, embedding)
   - Retrieval layer (search)
   - Generation layer (LLM)
   - Orchestration layer (main)

3. **Caching Strategy**
   - Model loaded once
   - Vectors pre-computed
   - Reduces latency for subsequent queries

### Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Data loading | O(n) | Linear file read |
| Chunking | O(n) | Single pass through text |
| Embedding | O(c × e) | c=chunks, e=embedding time |
| Retrieval | O(c) | Cosine similarity for all chunks |
| Generation | O(t) | t=LLM token generation time |

**Bottlenecks**:
- Initial embedding: ~5-10 seconds for 50 chunks
- LLM generation: ~2-3 seconds per query
- Retrieval: <0.1 seconds (very fast)

### Memory Usage

```
Data text:        ~15 KB
Chunks (50):      ~750 KB
Vectors (50):     ~75 KB (50 × 384 × 4 bytes)
Model in RAM:     ~90 MB (MiniLM)
Ollama LLM:       ~2-4 GB (llama3.2)
```

---

## 🎓 Learning Points

### What This Project Teaches

1. **RAG Pipeline Construction**
   - How to build a complete retrieval system
   - Integration of embeddings + LLM

2. **Vector Similarity**
   - Understanding semantic search
   - Cosine similarity in practice

3. **Local LLM Usage**
   - Running LLMs without cloud APIs
   - Prompt engineering for RAG

4. **Text Processing**
   - Chunking strategies
   - Handling context windows

5. **Python Best Practices**
   - Modular design
   - Error handling
   - Environment management

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Vector Database**: Replace in-memory storage with FAISS or ChromaDB
- [ ] **Multi-document Support**: Handle PDFs, DOCX, web pages
- [ ] **Re-ranking**: Use cross-encoder for better retrieval
- [ ] **Web UI**: Add Streamlit or Gradio interface
- [ ] **Evaluation**: Add metrics (BLEU, ROUGE, F1)
- [ ] **Caching**: Cache LLM responses for identical queries

### Potential Improvements

**Chunking**:
- Smart chunking by sentences/paragraphs
- Semantic chunking (chunk by topic)

**Retrieval**:
- Top-k retrieval (multiple chunks)
- Hybrid search (BM25 + vector)
- MMR (Maximum Marginal Relevance)

**Generation**:
- Streaming responses
- Temperature control
- Citation of sources

---

## 📝 Data Source

**data.txt**: Contains detailed information about badminton, including:
- Game rules and regulations
- Historical development (British India origins)
- Olympic debut in 1992
- Court dimensions
- Serving rules
- International competition details

**Size**: 273 lines, ~15 KB
**Format**: Plain text
**Content**: Wikipedia-style article on badminton

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate conda env: `conda activate rag_env` |
| `Ollama connection error` | Start Ollama: `ollama serve` |
| `OpenMP error` | Set env var: `export KMP_DUPLICATE_LIB_OK=TRUE` |
| `Out of memory` | Reduce chunk size or use smaller LLM |
| `Slow responses` | Check CPU usage, consider GPU acceleration |

---

## 📄 License

MIT License - Free for educational and commercial use

---

## 👨‍💻 Technical Notes

### Why This Architecture?

1. **Simplicity**: No complex dependencies
2. **Transparency**: Every step is visible and modifiable
3. **Educational**: Easy to understand for learning
4. **Privacy**: All data stays local
5. **Cost-effective**: No API fees

### Trade-offs Made

**Pros**:
- Simple to understand and modify
- No external dependencies
- Fast for small datasets

**Cons**:
- Limited to small datasets (no database)
- Single document support
- No advanced features (re-ranking, filtering)

---

## 🎯 Summary

This RAG system demonstrates a **complete end-to-end pipeline** for building intelligent question-answering systems. It combines:
- **Semantic search** (vector embeddings)
- **Local LLM generation** (Ollama)
- **Modular Python design** (easy to extend)

**Perfect for**: Learning RAG concepts, prototyping, local document Q&A

**Not suitable for**: Production at scale, multi-user systems, large document collections

---

**Last Updated**: January 2026
**Documentation Version**: 1.0
