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

This is a **Corrective RAG (CRAG)** system built in Python that enables intelligent question-answering over documents. The system combines **hybrid retrieval** (vector + BM25), **retrieval evaluation**, and **grounded generation** with answer validation to provide accurate, context-aware answers while reducing hallucinations.

### What is RAG?
RAG is a technique that enhances Large Language Model (LLM) responses by:
1. **Retrieving** relevant context from a knowledge base
2. **Augmenting** the user's query with this context
3. **Generating** an answer based on the retrieved information

### Key Features
- ✅ **Hybrid retrieval**: Vector similarity + BM25 keyword search
- ✅ **Retrieval evaluation**: Classifies relevance before generation
- ✅ **Corrective retry**: Query rewrite + rerank on weak retrieval
- ✅ **Grounded answers**: Enforced no-hallucination fallback
- ✅ **Answer validation**: Flags unsupported claims and regenerates
- ✅ **Modular design**: Each component is independent and testable
- ✅ **Efficient chunking**: Sliding window approach with overlap for better context
- ✅ **Conversational**: Interactive chat interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CORRECTIVE RAG PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA INGESTION                                           │
│     └─> Read PDF / TXT / CSV / YouTube                      │
│                                                              │
│  2. CHUNKING                                                 │
│     └─> Split into overlapping segments                     │
│                                                              │
│  3. EMBEDDING                                                │
│     └─> Convert chunks to 384-dim vectors                   │
│                                                              │
│  4. HYBRID RETRIEVAL                                         │
│     └─> Vector + BM25 keyword search                        │
│                                                              │
│  5. RETRIEVAL EVALUATION                                     │
│     └─> relevant / partially relevant / irrelevant          │
│                                                              │
│  6. CORRECTIVE RETRY                                         │
│     └─> Query rewrite + rerank + re-eval                    │
│                                                              │
│  7. GENERATION                                               │
│     └─> Strict grounded prompt                              │
│                                                              │
│  8. ANSWER VALIDATION                                        │
│     └─> Detect unsupported claims                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Breakdown

### Corrective RAG Modules

- **crag/config.py**: Centralized CRAG thresholds and feature flags
- **crag/hybrid_retrieval.py**: Vector + BM25 retrieval with score fusion
- **crag/retrieval_evaluator.py**: Retrieval quality classification
- **crag/query_rewriter.py**: Automatic query rewriting for weak queries
- **crag/reranker.py**: Lexical overlap reranking
- **crag/context_filter.py**: Deduplication and confidence filtering
- **crag/validator.py**: Answer grounding validation
- **crag/controller.py**: Corrective RAG orchestration
- **crag/metrics.py**: Retrieval/grounding evaluation utilities

### Retrieval Evaluation and Correction

1. **Hybrid Retrieval**: Vector similarity + BM25 keyword search returns top-k candidates.
2. **Evaluation**: Uses raw cosine similarity thresholds to classify relevance.
3. **Correction**: If relevance is weak, the query is rewritten and retrieval is retried.
4. **Reranking**: Combines semantic scores with lexical overlap for better ordering.
5. **Filtering**: Deduplicates and drops low-confidence chunks before generation.

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

### 2. **csv_ingestion.py** - CSV Ingestion Module
**Purpose**: Convert structured CSV rows into searchable text

**Highlights**:
- Preserves column names as labels
- Converts each row into a single line of text
- Enables natural-language queries over tabular data

---

### 3. **chunking.py** - Text Chunking Module
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
  - `chunk_size`: Number of characters per chunk (default: ~600, recommended 500–1000)
  - `overlap`: Number of overlapping characters between chunks (default: ~120, recommended 100–200)
  
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

### 4. **embedding.py** - Vector Embedding Module
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

### 5. **retrieval.py** - Smart Retrieval Module
**Purpose**: Find the most relevant text chunk for a given query

**Code Analysis**:
```python
from sentence_transformers import util
import torch

def search_best_chunks(query, model, db_vectors, chunks, k=3):
    # 1. Encode the query
    query_vector = model.encode(query)

    # 2. Calculate similarity scores
    scores = util.cos_sim(query_vector, db_vectors)

    # 3. Find top-k matches
    top_results = torch.topk(scores, k=k)

    # 4. Return results
    return [
        {"text": chunks[idx.item()], "score": top_results.values[0][i].item()}
        for i, idx in enumerate(top_results.indices[0])
    ]
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

### 6. **retreival.py** (Legacy Version)
**Note**: This is an older version with different architecture

**Key Difference**:
- Loads data/model at module level (once during import)
- `search_best_chunk()` takes only the query as parameter
- Less flexible than `retrieval.py` (can't reuse model for different datasets)

**Status**: Likely deprecated in favor of `retrieval.py`

---

### 7. **generation.py** - Answer Generation Module
**Purpose**: Generate natural language answers using a local LLM

**Code Analysis**:
```python
import os
import requests

def generate_answer(query, context):
    prompt = f"""
    You are a grounded assistant. Answer ONLY from the context.
    If the answer is not present, respond with:
    "The uploaded document does not contain enough information to answer this question."

    Context: {context}

    Question: {query}
    """

    api_key = os.getenv("GEMINI_API_KEY")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    response = requests.post(url, json=payload)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
```

**Components**:

1. **Prompt Engineering**:
   - Critical instruction: "Answer ONLY from the provided context"
   - Prevents hallucination (making up information)
   - Fallback: explicit grounded message when context is missing

2. **Gemini Integration**:
   - Uses `gemini-1.5-flash` via HTTPS
   - API key from `GEMINI_API_KEY`
   - Non-streaming mode for simplicity

3. **Error Handling**:
   - Connection errors surfaced to the UI
   - Returns fallback message when generation fails

---

### 8. **main.py** - Orchestration Module
**Purpose**: Tie everything together into an interactive application

**Code Analysis**:
```python
def start_app():
    # INITIALIZATION PHASE (Run Once)
    raw_text = reading_data("data.txt")
    text_chunks = get_chunks(raw_text, 600, 120)
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
│ 5. LLM GENERATION (Gemini)               │
│    "Badminton joined the Olympics in    │
│     1992 as a Summer Olympic sport."     │
└─────────────────────────────────────────┘
              ↓
         Answer to User
```

### Corrective RAG Flow (CRAG)

```
User Query
   ↓
Hybrid Retrieval (Vector + BM25)
   ↓
Retrieval Evaluation (relevant / partial / irrelevant)
   ↓
[Relevant]
   → Context Filtering
   → Generation (strict prompt)
   → Answer Validation
   → Final Response

[Irrelevant or Partial]
   → Query Rewrite
   → Hybrid Retrieval Retry
   → Rerank + Re-evaluate
   → Generation + Validation
```

---

## 🛠️ Technical Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 2.7.0 | Text embedding and similarity |
| `torch` | 2.6.0 | Tensor operations |
| `requests` | 2.32.3 | HTTP communication with Gemini |
| `scikit-learn` | 1.4.2 | ML utilities (future use) |

### External Services

| Service | Purpose |
|---------|---------|
| **Gemini API** | Grounded LLM inference |
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
pip install -r requirements.txt

# 3. Set Gemini API key
export GEMINI_API_KEY="your_api_key"
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

# Check Gemini API key
echo $GEMINI_API_KEY
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
Gemini API:       External inference (no local LLM memory)
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

3. **API-based LLM Usage**
   - Using Gemini for grounded generation
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
| `Gemini API error` | Verify `GEMINI_API_KEY` |
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
4. **Privacy**: Only retrieved chunks are sent to Gemini
5. **Cost-aware**: API usage depends on query volume

### Trade-offs Made

**Pros**:
- Simple to understand and modify
- Clear separation of ingestion/retrieval/generation
- Fast for small datasets

**Cons**:
- Limited to small datasets (no database)
- Requires external Gemini API access
- No advanced features (re-ranking, filtering)

---

## 🎯 Summary

This RAG system demonstrates a **complete end-to-end pipeline** for building intelligent question-answering systems. It combines:
- **Semantic search** (vector embeddings)
- **Gemini generation** (grounded answers)
- **Modular Python design** (easy to extend)

**Perfect for**: Learning RAG concepts, prototyping, local document Q&A

**Not suitable for**: Production at scale, multi-user systems, large document collections

---

**Last Updated**: January 2026
**Documentation Version**: 1.0
