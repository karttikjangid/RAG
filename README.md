# 🎓 LecturMate - NotebookLM-Style RAG Application

> A powerful, Gemini-powered Retrieval-Augmented Generation system with a beautiful Streamlit web interface for querying your documents and YouTube videos using AI.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Overview

**LecturMate** is a complete RAG (Retrieval-Augmented Generation) system that lets you chat with PDFs, TXT files, CSVs, and video transcripts using **Gemini** for grounded answers. Built with a modern, calm green UI inspired by contemporary design principles.

### Key Capabilities

- 📄 **Multi-Source Ingestion**: PDFs, TXT files, CSVs, and YouTube videos
- 🧠 **Smart Chunking**: Sliding window approach with configurable overlap
- 🔍 **Semantic Search**: Find relevant context using vector embeddings
- 🗂️ **Vector Store**: In-memory embedding matrix for fast similarity search
- 💬 **AI-Powered Answers**: Gemini API (gemini-1.5-flash) with grounded responses
- 🎨 **Beautiful UI**: Modern Streamlit interface with light green theme
- ⚡ **Performance Optimized**: Comprehensive caching for lightning-fast responses
- 🔒 **Grounded RAG**: Answers are restricted to retrieved context with explicit no-hallucination fallback


## 📸 Screenshots (Placeholders)

- Web interface overview *(add screenshot here)*
- Chat interface with grounded answer *(add screenshot here)*
- Mobile responsive layout *(add screenshot here)*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LecturMate RAG Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INGESTION       →  2. CHUNKING    →  3. EMBEDDING     │
│  ┌──────────────┐     ┌───────────┐     ┌──────────────┐  │
│  │ PDF/TXT/CSV │      │ Sliding   │     │ Sentence     │  │
│  │ YouTube URLs│  →   │ Window    │  →  │ Transformers │  │
│  │            │       │ (600/120) │     │ (384-dim)    │  │
│  └──────────────┘     └───────────┘     └──────────────┘  │
│         ↓                    ↓                   ↓         │
│  4. RETRIEVAL       ←  5. GENERATION   ←  User Query      │
│  ┌──────────────┐     ┌───────────────────────────┐       │
│  │ Cosine       │     │ Gemini (gemini-1.5-flash)│       │
│  │ Similarity   │  ←  │ + Retrieved Context       │       │
│  │ Top-k=3      │     │ = Grounded Answer         │       │
│  └──────────────┘     └───────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Grounding & Hallucination Prevention

LecturMate strictly grounds answers in retrieved context:

- **Prompt guardrails** instruct Gemini to use only provided chunks.
- **Similarity threshold** blocks low-confidence retrievals.
- **Fallback message** when context is insufficient:
  *"The uploaded document does not contain enough information to answer this question."*

---

## 🧩 Chunking Strategy

- **Splitter**: Existing sliding-window chunking in `chunking.py`
- **Chunk size**: ~600 characters (recommended 500–1000)
- **Overlap**: ~120 characters (recommended 100–200)
- **Why overlap**: Preserves context across chunk boundaries

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
│   ├── csv_ingestion.py          # CSV ingestion
│   ├── pdf_ingestion.py          # PDF extraction (pypdf)
│   ├── youtube_ingestion.py      # YouTube transcript fetching
│   ├── chunking.py               # Sliding window text splitting
│   ├── embedding.py              # Vector embeddings (all-MiniLM-L6-v2)
│   ├── retrieval.py              # Semantic search with cosine similarity
│   └── generation.py             # LLM answer generation (Gemini)
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
- **Gemini API key** (set in `GEMINI_API_KEY`)
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
   pip install -r requirements.txt
   ```
 
4. **Set environment variables:**
   ```bash
   export GEMINI_API_KEY="your_api_key"
   ```
   You can also copy `.env.example` to `.env` and load it in your environment.

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
   - Upload **PDF, TXT, or CSV** files
   - Click **"🎥 YouTube"** tab to paste video URLs (optional)
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

## 🚀 Deployment

LecturMate is deployable on Render or Railway:

1. Add a `GEMINI_API_KEY` environment variable in your hosting provider.
2. Install dependencies from `requirements.txt`.
3. Set the start command:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Deploy and open the generated public URL.

---

## 🧪 Example Queries

- "Summarize the key findings from the uploaded PDF."
- "What are the main columns and trends in the CSV data?"
- "List the key definitions mentioned in the TXT notes."

---

## ⚙️ Configuration

### Chunking Parameters

Adjust in `chunking.py` or pass to functions:
```python
chunk_size = 600   # Characters per chunk (500–1000 recommended)
overlap = 120      # Overlapping characters (100–200 recommended)
```

### Embedding Model

Change in `embedding.py`:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
# Alternatives: 'all-mpnet-base-v2' (768-dim, better quality, slower)
```

### LLM Model

Gemini is configured in `generation.py`:
```python
GEMINI_MODEL = "gemini-1.5-flash"
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
2. **Vector DB**: In-memory embedding matrix stored in session state
3. Cosine similarity computed against all chunks
4. Top-3 most relevant chunks selected
5. Combined as context for Gemini

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
from csv_ingestion import get_csv_text
from youtube_ingestion import get_youtube_transcript
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import generate_answer

# 1. Ingest data
pdf_text = get_pdf_text("research.pdf")
csv_text = get_csv_text("metrics.csv")
yt_text = get_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
combined_text = pdf_text + "\n\n" + csv_text + "\n\n" + yt_text

# 2. Chunk the text
chunks = get_chunks(combined_text, chunk_size=600, overlap=120)

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

# CSV Ingestion
from csv_ingestion import get_csv_text
text = get_csv_text("data.csv")  # Returns: str

# Chunking
from chunking import get_chunks
chunks = get_chunks(text, chunk_size=600, overlap=120)  # Returns: list[str]

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

**2. Gemini API errors**
```bash
# Verify the API key is set
echo $GEMINI_API_KEY
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
4. **LLM Model**: `gemini-1.5-flash` is fast; use `gemini-2.0-flash` for higher quality

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
- **Gemini API** for grounded LLM inference
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
