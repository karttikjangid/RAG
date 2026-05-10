# RAG System - Complete Data Ingestion Guide

## 📚 Overview

The RAG system now supports **four different data sources**, each with its own specialized ingestion module. All modules output clean text ready for the RAG pipeline.

---

## 🔧 Available Ingestion Modules

### 1️⃣ Text File Ingestion
**File**: `data_ingestion.py`  
**Function**: `reading_data(file_path)`  
**Use Case**: Plain text files, markdown, code files

```python
from data_ingestion import reading_data

text = reading_data("notes.txt")
```

**Features**:
- ✅ UTF-8 encoding support
- ✅ Simple and fast
- ✅ Any text-based file format

---

### 2️⃣ CSV File Ingestion
**File**: `csv_ingestion.py`  
**Function**: `get_csv_text(file_path)`  
**Use Case**: Tables, datasets, structured reports

```python
from csv_ingestion import get_csv_text

text = get_csv_text("report.csv")
```

**Features**:
- ✅ Preserves column context (e.g., `Column: value`)
- ✅ Converts rows into searchable text
- ✅ Handles missing values gracefully

---

### 3️⃣ YouTube Video Transcripts
**File**: `youtube_ingestion.py`  
**Function**: `get_youtube_transcript(url)`  
**Use Case**: Educational videos, tutorials, lectures

```python
from youtube_ingestion import get_youtube_transcript

url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
text = get_youtube_transcript(url)
```

**Features**:
- ✅ Automatic subtitle extraction
- ✅ Timestamp removal
- ✅ Support for auto-generated & manual captions
- ✅ Multiple language support
- 📦 Library: `youtube-transcript-api`

---

### 4️⃣ PDF Documents
**File**: `pdf_ingestion.py`  
**Function**: `get_pdf_text(file_path)`  
**Use Case**: Research papers, books, reports, resumes

```python
from pdf_ingestion import get_pdf_text

text = get_pdf_text("research_paper.pdf")
```

**Features**:
- ✅ Multi-page extraction
- ✅ Automatic text cleaning
- ✅ Progress indicators
- ✅ Comprehensive error handling
- 📦 Library: `pypdf`

---

## 🎯 Quick Comparison

| Feature | Text Files | CSV Files | YouTube | PDF |
|---------|-----------|-----------|---------|-----|
| **Module** | `data_ingestion.py` | `csv_ingestion.py` | `youtube_ingestion.py` | `pdf_ingestion.py` |
| **Input** | File path | File path | YouTube URL | File path |
| **Library** | Built-in | Built-in (`csv`) | `youtube-transcript-api` | `pypdf` |
| **Speed** | ⚡ Instant | ⚡ Instant | 🔄 2-5 sec | ⚡ < 1 sec |
| **Pages** | N/A | Rows | Single video | Multiple pages |
| **Error Handling** | Basic | Advanced | Advanced | Advanced |
| **Output Format** | Raw text | Row-labeled text | Clean transcript | Combined pages |

---

## 🚀 Complete RAG Pipeline Integration

### Multi-Source RAG Example

```python
from data_ingestion import reading_data
from csv_ingestion import get_csv_text
from youtube_ingestion import get_youtube_transcript
from pdf_ingestion import get_pdf_text
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import generate_answer

# 1. Gather data from multiple sources
text_data = reading_data("notes.txt")
csv_data = get_csv_text("metrics.csv")
youtube_data = get_youtube_transcript("https://youtube.com/watch?v=...")
pdf_data = get_pdf_text("research.pdf")

# 2. Combine all sources
all_text = text_data + " " + csv_data + " " + youtube_data + " " + pdf_data

# 3. Process through RAG pipeline
chunks = get_chunks(all_text, chunk_size=600, overlap=120)
vectors, model = vector_embedding(chunks)

# 4. Query the system
query = "What is RAG?"
results = search_best_chunks(query, model, vectors, chunks, k=3)

# 5. Generate answer
answer = generate_answer(query, results[0]['text'])
print(answer)
```

---

## 📦 Installation Requirements

```bash
# Activate virtual environment
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt

# Set Gemini API key
export GEMINI_API_KEY="your_api_key"
```

---

## ✅ Testing Each Module

### Test Text Ingestion
```bash
python data_ingestion.py
# Expected: Character count of data.txt
```

### Test YouTube Ingestion
```bash
python youtube_ingestion.py
# Expected: Transcript from first YouTube video
```

### Test PDF Ingestion
```bash
python pdf_ingestion.py
# Expected: Text from sample.pdf (3 pages)
```

### Test Full Integration
```bash
python pdf_rag_example.py
# Expected: Complete RAG workflow with Q&A
```

---

## 🎓 Module Architecture

```
RAG_SYSTEM/
│
├── 📥 DATA INGESTION LAYER
│   ├── data_ingestion.py         (Text files)
│   ├── csv_ingestion.py          (CSV files)
│   ├── youtube_ingestion.py      (YouTube videos)
│   └── pdf_ingestion.py          (PDF documents)
│
├── ⚙️ PROCESSING LAYER
│   ├── chunking.py               (Text splitting)
│   └── embedding.py              (Vector creation)
│
├── 🔍 RETRIEVAL LAYER
│   └── retrieval.py              (Semantic search)
│
├── 🤖 GENERATION LAYER
│   └── generation.py             (LLM answers)
│
└── 🎯 ORCHESTRATION
    └── main.py                   (Complete pipeline)
```

---

## 📊 Real-World Use Cases

### Use Case 1: Academic Research
```python
# Combine research papers + lecture videos
papers = get_pdf_text("ml_paper.pdf")
lecture = get_youtube_transcript("stanford_lecture_url")
combined = papers + " " + lecture

# Build knowledge base
chunks = get_chunks(combined, 300, 75)
vectors, model = vector_embedding(chunks)

# Ask questions
answer = search_best_chunks("What is gradient descent?", model, vectors, chunks)
```

### Use Case 2: Company Documentation
```python
# Aggregate all company knowledge
handbook = get_pdf_text("employee_handbook.pdf")
training = get_youtube_transcript("onboarding_video")
policies = reading_data("policies.txt")

# Create searchable knowledge base
all_docs = handbook + " " + training + " " + policies
# ... process through RAG ...
```

### Use Case 3: Learning Assistant
```python
# Build personal study guide
textbook = get_pdf_text("chapter5.pdf")
tutorial = get_youtube_transcript("khan_academy_url")
notes = reading_data("my_notes.txt")

# RAG-powered study assistant
knowledge_base = textbook + " " + tutorial + " " + notes
# ... answer study questions ...
```

---

## 🔍 Error Handling Summary

All three modules implement consistent error handling:

| Error Type | Behavior |
|------------|----------|
| File not found | Returns error message string |
| Invalid format | Returns error message string |
| Network issues | Returns error message string |
| Empty content | Returns empty string or message |
| No crash | ✅ Always returns a string |

**Check for errors**:
```python
result = get_pdf_text("file.pdf")
if result.startswith("❌"):
    print("Error occurred:", result)
else:
    print("Success!")
```

---

## 📝 Best Practices

### 1. Chunk Size Selection
- **Small files** (< 5 pages): chunk_size=100-150
- **Medium files** (5-20 pages): chunk_size=200-300
- **Large files** (> 20 pages): chunk_size=300-500

### 2. Overlap Strategy
- **High precision needed**: overlap=50-100
- **General use**: overlap=20-50
- **Fast processing**: overlap=10-20

### 3. Multi-Source Integration
```python
# Weight different sources if needed
pdf_text = get_pdf_text("primary.pdf")
youtube_text = get_youtube_transcript(url)

# Primary source repeated for higher weight
combined = pdf_text + " " + pdf_text + " " + youtube_text
```

### 4. Pre-processing
```python
# Clean extracted text before chunking
text = get_pdf_text("document.pdf")
text = text.replace("\n\n", " ")  # Remove extra newlines
text = ' '.join(text.split())      # Normalize whitespace
```

---

## 🎯 Quick Reference Commands

```bash
# Create sample PDF for testing
python create_sample_pdf.py

# Quick PDF test
python quick_test_pdf.py

# Test error handling
python test_pdf_errors.py

# Full RAG demo with PDF
python pdf_rag_example.py

# Full RAG demo with YouTube
python youtube_rag_example.py

# Run main RAG application
python main.py
```

---

## 📚 Documentation Files

- **RAG_SYSTEM_DOCUMENTATION.md** - Complete system overview
- **YOUTUBE_INGESTION_DOCS.md** - YouTube module details
- **PDF_INGESTION_DOCS.md** - PDF module details
- **README.md** - Original project documentation
- **This file** - Multi-source ingestion guide

---

## 🚀 Next Steps

With all three ingestion modules complete, you can now:

1. ✅ Ingest data from **any source** (text, video, PDF)
2. ✅ Build **multi-modal knowledge bases**
3. ✅ Create **domain-specific RAG systems**
4. ✅ Scale to **production workloads**

### Future Enhancements
- [ ] Web scraping module (BeautifulSoup)
- [ ] Audio transcription (Whisper API)
- [ ] Database integration (SQL/NoSQL)
- [ ] Cloud storage support (S3, GCS)
- [ ] Real-time data streams
- [ ] Document OCR for scanned PDFs

---

**Status**: 🎉 All Core Ingestion Modules Complete!  
**Last Updated**: January 7, 2026  
**Ready for**: Production Use
