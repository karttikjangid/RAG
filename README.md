# 🎓 LecturMate - Multi-Modal RAG Application

> A powerful, local-first Retrieval-Augmented Generation system with a beautiful Streamlit web interface for querying your documents and YouTube videos using AI.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Overview

**LecturMate** is a complete RAG (Retrieval-Augmented Generation) system that lets you chat with your documents and video transcripts using AI - entirely offline and free. Built with a modern, calm green UI inspired by contemporary design principles.

### Key Capabilities

- 📄 **Multi-Source Ingestion**: PDFs, YouTube videos, and text files
- 🧠 **Smart Chunking**: Sliding window approach with configurable overlap
- 🔍 **Semantic Search**: Find relevant context using vector embeddings
- 💬 **AI-Powered Answers**: Local LLM generation with Ollama (no API costs!)
- 🎨 **Beautiful UI**: Modern Streamlit interface with light green theme
- ⚡ **Performance Optimized**: Comprehensive caching for lightning-fast responses
- 🔒 **Privacy First**: Everything runs locally - your data never leaves your machine


## 📸 Screenshots

### Web Interface (app.py)
- Modern chat interface with message history
- Tabbed sidebar for PDF and YouTube uploads
- Real-time source management with delete functionality
- Auto-processing on source addition
- Calm light green aesthetic with perfect readability

### CLI Interface (main.py)
- Interactive terminal-based question answering
- Multi-source selection and processing
- Useful for server environments or scripting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LecturMate RAG Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INGESTION       →  2. CHUNKING    →  3. EMBEDDING     │
│  ┌──────────────┐     ┌───────────┐     ┌──────────────┐  │
│  │ PDF Files    │     │ Sliding   │     │ Sentence     │  │
│  │ YouTube URLs │  →  │ Window    │  →  │ Transformers │  │
│  │ Text Files   │     │ (500/100) │     │ (384-dim)    │  │
│  └──────────────┘     └───────────┘     └──────────────┘  │
│         ↓                    ↓                   ↓         │
│  4. RETRIEVAL       ←  5. GENERATION   ←  User Query      │
│  ┌──────────────┐     ┌───────────────────────────┐       │
│  │ Cosine       │     │ Ollama LLM (llama3.2)    │       │
│  │ Similarity   │  ←  │ + Retrieved Context       │       │
│  │ Top-k=3      │     │ = Accurate Answer         │       │
│  └──────────────┘     └───────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG/
├── 🌐 Web Interface
│   └── app.py                    # Streamlit web app with caching & UI
│
├── 🖥️  CLI Interface
│   └── main.py                   # Terminal-based interactive RAG
│
├── 🔧 Core Modules
│   ├── data_ingestion.py         # Text file ingestion
│   ├── pdf_ingestion.py          # PDF extraction (pypdf)
│   ├── youtube_ingestion.py      # YouTube transcript fetching
│   ├── chunking.py               # Sliding window text splitting
│   ├── embedding.py              # Vector embeddings (all-MiniLM-L6-v2)
│   ├── retrieval.py              # Semantic search with cosine similarity
│   └── generation.py             # LLM answer generation (Ollama)
│
├── 📚 Documentation
│   ├── docs/
│   │   ├── RAG_SYSTEM_DOCUMENTATION.md
│   │   ├── COMPLETE_INGESTION_GUIDE.md
│   │   ├── YOUTUBE_INGESTION_DOCS.md
│   │   └── PDF_INGESTION_DOCS.md
│   └── README.md                 # You are here!
│
├── 🧪 Tests
│   └── tests/
│       ├── test_real_rag.py
│       └── README.md
│
└── 📄 Data
    └── data.txt                  # Sample data
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **Ollama** installed and running ([Download here](https://ollama.ai))
- **4GB+ RAM** recommended for embedding model

### Installation

1. **Clone or download the repository**

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit==1.52.2
   pip install sentence-transformers
   pip install scikit-learn
   pip install youtube-transcript-api
   pip install pypdf
   pip install torch
   ```

4. **Install and configure Ollama:**
   ```bash
   # Download from https://ollama.ai, then:
   ollama pull llama3.2
   ollama serve  # Keep this running in a separate terminal
   ```

### Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` 🎉

### Alternative: CLI Mode

For terminal-based usage:
```bash
python main.py
```

---

## 💡 Usage Guide

### Web Interface (Recommended)

1. **Add Sources:**
   - Click **"📄 PDF Upload"** tab to upload PDF documents
   - Click **"🎥 YouTube"** tab to paste video URLs
   - Sources are auto-processed and displayed in the sidebar

2. **Ask Questions:**
   - Type your question in the chat input at the bottom
   - Press Enter or click Send
   - Get AI-powered answers with context from your sources

3. **Manage Sources:**
   - View all added sources in the sidebar with metadata
   - Click **"Delete Source"** to remove unwanted sources
   - Vector database updates automatically

### CLI Interface

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
Enter PDF file path: research_paper.pdf

✅ Successfully loaded 61,447 characters
✅ Created 410 chunks (500 chars each, 100 overlap)
✅ Generated 410 vector embeddings

❓ Your question: What is the main conclusion?
🔍 Searching knowledge base...
💬 Answer: Based on the research paper, the main conclusion is...

❓ Your question (or 'quit'): _
```

---

## ⚙️ Configuration

### Chunking Parameters

Adjust in `chunking.py` or pass to functions:
```python
chunk_size = 500   # Characters per chunk
overlap = 100      # Overlapping characters (improves context continuity)
```

### Embedding Model

Change in `embedding.py`:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
# Alternatives: 'all-mpnet-base-v2' (768-dim, better quality, slower)
```

### LLM Model

Update in `generation.py`:
```python
model = "llama3.2"  # Default
# Alternatives: "mistral", "llama2", "phi3", etc.
```

### Retrieval Settings

Modify `k` parameter in `retrieval.py`:
```python
k = 3  # Number of top chunks to retrieve
# Higher k = more context but slower generation
```

---

## 🎨 Features Breakdown

### Multi-Source Ingestion
- **PDFs**: Extracts text from all pages, handles complex layouts
- **YouTube**: Auto-fetches official transcripts, cleans timestamps
- **Text Files**: Direct .txt file loading

### Smart Caching System
- **Model Caching**: Loads embedding model once with `@st.cache_resource`
- **PDF Caching**: Hash-based caching - same PDF won't be re-processed
- **YouTube Caching**: 1-hour TTL cache for transcripts
- **Embedding Caching**: Stores computed vectors to avoid recomputation
- **Result**: 10-100x faster on repeated queries!

### Retrieval Mechanism
1. User query → Embedded to 384-dim vector
2. Cosine similarity computed against all chunks
3. Top-3 most relevant chunks selected
4. Combined as context for LLM

### UI/UX Design
- **Light Green Theme**: Calming #66bb6a, #4caf50 color palette
- **Manrope Font**: Professional, readable typeface
- **Responsive Layout**: Works on desktop and tablets
- **Message History**: Persistent chat across sessions
- **Source Cards**: Visual display of added documents with metadata

---

## 🔧 Programmatic Usage

For developers who want to integrate LecturMate into their own projects:

### Basic Pipeline

```python
from pdf_ingestion import get_pdf_text
from youtube_ingestion import get_youtube_transcript
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import generate_answer

# 1. Ingest data
pdf_text = get_pdf_text("research.pdf")
yt_text = get_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
combined_text = pdf_text + "\n\n" + yt_text

# 2. Chunk the text
chunks = get_chunks(combined_text, chunk_size=500, overlap=100)

# 3. Create embeddings
vectors, model = vector_embedding(chunks)

# 4. Query the system
query = "What is the main topic discussed?"
results = search_best_chunks(query, model, db_vectors=vectors, chunks=chunks, k=3)

# 5. Generate answer
context = results[0]['text']  # Top result
answer = generate_answer(query, context)
print(answer)
```

### Individual Module Usage

```python
# PDF Ingestion
from pdf_ingestion import get_pdf_text
text = get_pdf_text("document.pdf")  # Returns: str

# YouTube Ingestion
from youtube_ingestion import get_youtube_transcript
text = get_youtube_transcript("https://youtu.be/VIDEO_ID")  # Returns: str

# Text File Ingestion
from data_ingestion import reading_data
text = reading_data("data.txt")  # Returns: str

# Chunking
from chunking import get_chunks
chunks = get_chunks(text, chunk_size=500, overlap=100)  # Returns: list[str]

# Embedding
from embedding import vector_embedding
vectors, model = vector_embedding(chunks)  # Returns: (ndarray, SentenceTransformer)

# Retrieval
from retrieval import search_best_chunks
results = search_best_chunks(
    query="your question",
    model=model,
    db_vectors=vectors,
    chunks=chunks,
    k=3  # number of results
)  # Returns: list[dict] with 'text', 'score', 'index'

# Generation
from generation import generate_answer
answer = generate_answer(query="question", context="retrieved context")  # Returns: str
```

---

## 📚 Documentation

Comprehensive guides available in the [docs/](docs/) directory:

| Document | Description |
|----------|-------------|
| [RAG System Documentation](docs/RAG_SYSTEM_DOCUMENTATION.md) | Complete system architecture and design |
| [Ingestion Guide](docs/COMPLETE_INGESTION_GUIDE.md) | Multi-source data ingestion details |
| [YouTube Module](docs/YOUTUBE_INGESTION_DOCS.md) | YouTube transcript extraction guide |
| [PDF Module](docs/PDF_INGESTION_DOCS.md) | PDF processing implementation |

---

## 🐛 Troubleshooting

### Common Issues

**1. Streamlit won't start**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Check Streamlit installation
pip install --upgrade streamlit
```

**2. Ollama connection error**
```bash
# Check if Ollama is running
curl http://localhost:11434

# Start Ollama service
ollama serve

# Verify model is downloaded
ollama list
```

**3. PyTorch/CUDA issues**
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Reinstall PyTorch (CPU version)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**4. PDF extraction errors**
```bash
# Reinstall pypdf
pip uninstall pypdf
pip install pypdf
```

**5. YouTube transcript not found**
- Not all videos have transcripts (auto-generated or manual)
- Try a different video or enable captions on YouTube first
- Check video URL is correct and public

**6. Memory issues**
```python
# Reduce chunk size to use less memory
chunks = get_chunks(text, chunk_size=300, overlap=50)  # Smaller chunks

# Or use a smaller embedding model in embedding.py
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim (current)
# vs 'all-mpnet-base-v2'  # 768-dim (larger)
```

---

## 🚀 Performance Tips

1. **Caching**: The web app automatically caches models, PDFs, and embeddings - subsequent runs are much faster
2. **Chunk Size**: Larger chunks (500-1000) = better context but slower; smaller chunks (200-300) = faster but may miss context
3. **Top-k**: Retrieving fewer chunks (k=2-3) is faster than k=5-10
4. **LLM Model**: Smaller models like `phi3` are faster than `llama3.2` but may have lower quality

---

## 🗺️ Roadmap

Future enhancements planned:

- [ ] Vector database integration (FAISS/ChromaDB) for persistent storage
- [ ] Multi-document comparison and cross-referencing
- [ ] Export chat history to PDF/Markdown
- [ ] Dark mode toggle
- [ ] Support for more file types (DOCX, PPTX, HTML)
- [ ] Advanced filters (date ranges, source types)
- [ ] Audio file transcription (Whisper integration)
- [ ] Evaluation metrics (answer quality, retrieval accuracy)
- [ ] Multi-language support
- [ ] Hosted version with authentication

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Sentence Transformers** for efficient embedding models
- **Ollama** for local LLM inference
- **Streamlit** for the beautiful web framework
- **YouTube Transcript API** for easy transcript access
- **pypdf** for PDF text extraction

---

## 📧 Contact

Built with ❤️ for learning and exploring RAG systems.

**Author**: Kartik Jangid  
**Repository**: [github.com/karttikjangid/RAG](https://github.com/karttikjangid/RAG)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Happy Learning! 🎓**
