# main.py
import time
import sys
from data_ingestion import reading_data
from youtube_ingestion import get_youtube_transcript
from pdf_ingestion import get_pdf_text
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import generate_answer

def start_app():
    print("\n" + "=" * 70)
    print("🚀 RAG SYSTEM - MULTI-SOURCE DATA INGESTION")
    print("=" * 70)
    
    # Display Menu
    print("\n📚 SELECT DATA SOURCE:")
    print("   1. Text File (.txt)")
    print("   2. PDF Document (.pdf)")
    print("   3. YouTube Video (URL)")
    print()
    
    # Get user choice
    choice = input("Enter your choice (1-3): ").strip()
    
    # 1. Load Data Based on Choice
    print("\n--- 📥 LOADING DATA ---")
    raw_text = None
    
    if choice == "1":
        # Text File
        file_path = input("Enter text file path (or press Enter for 'data.txt'): ").strip()
        if not file_path:
            file_path = "data.txt"
        print(f"Loading: {file_path}")
        raw_text = reading_data(file_path)
        
    elif choice == "2":
        # PDF Document
        pdf_path = input("Enter PDF file path: ").strip()
        print(f"Loading: {pdf_path}")
        raw_text = get_pdf_text(pdf_path)
        
    elif choice == "3":
        # YouTube Video
        video_url = input("Enter YouTube URL: ").strip()
        print(f"Loading: {video_url}")
        raw_text = get_youtube_transcript(video_url)
        
    else:
        print("❌ Invalid choice! Please select 1, 2, or 3.")
        sys.exit(1)
    
    # Safety Check: Ensure we have valid text
    if not raw_text or raw_text.startswith("❌"):
        print("\n❌ ERROR: Failed to load data or data is empty!")
        if raw_text:
            print(f"   {raw_text}")
        print("   Please check your input and try again.")
        sys.exit(1)
    
    print(f"✅ Successfully loaded {len(raw_text)} characters")
    
    # 2. Chunk the Data
    print("\n--- ✂️  CHUNKING DATA ---")
    text_chunks = get_chunks(raw_text, 200, 50)
    print(f"   ✅ Created {len(text_chunks)} chunks")
    
    # 3. Embed (Create Vector Embeddings)
    print("\n--- 🧠 CREATING EMBEDDINGS ---")
    print("   Loading embedding model...")
    db_vectors, model = vector_embedding(text_chunks)
    print(f"   ✅ Generated {len(db_vectors)} vectors")
    
    print("\n" + "=" * 70)
    print("✅ SYSTEM READY - Ask your questions!")
    print("   Type 'exit' to quit")
    print("=" * 70)

    # 4. The Interactive Chat Loop
    while True:
        query = input("\n❓ Your question: ").strip()
        
        if query.lower() == "exit":
            print("\n👋 Goodbye! Thanks for using RAG System!")
            break
        
        if not query:
            continue
            
        start_ts = time.time()
        
        # A. RETRIEVAL - Find relevant context
        print(f"🔍 Searching knowledge base...")
        results = search_best_chunks(query, model, db_vectors, text_chunks, k=3)
        best_context = results[0]['text']
        score = results[0]['score']
        
        print(f"   📄 Best match (confidence: {score:.4f})")
        print(f"   \"{best_context[:150]}...\"")
        
        # B. GENERATION - Generate answer with LLM
        print("🤖 Generating answer...")
        try:
            answer = generate_answer(query, best_context)
            print(f"\n💬 Answer:\n{answer}")
        except Exception as e:
            print(f"\n⚠️  LLM not available. Here's the retrieved context:")
            print(f"{best_context[:500]}...")
        
        end_ts = time.time()
        print(f"\n⏱️  Time taken: {end_ts - start_ts:.2f}s")
        print("-" * 70)

if __name__ == "__main__":
    start_app()