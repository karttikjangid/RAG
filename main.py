# main.py
import time
import sys
from data_ingestion import reading_data
from csv_ingestion import get_csv_text
from youtube_ingestion import get_youtube_transcript
from pdf_ingestion import get_pdf_text
from chunking import get_chunks
from embedding import vector_embedding
from generation import NO_ANSWER_RESPONSE
from crag.config import CRAGConfig
from crag.controller import CorrectiveRAGController
from crag.hybrid_retrieval import HybridRetriever
from crag.logging_utils import get_logger


def start_app():
    print("\n" + "=" * 70)
    print("🚀 RAG SYSTEM - MULTI-SOURCE DATA AGGREGATOR")
    print("=" * 70)
    print("\n💡 TIP: Load multiple sources to create a comprehensive knowledge base!")
    print("   You can combine PDFs, YouTube videos, and text files.\n")
    
    # The Accumulator - Shopping Cart for data
    all_text = ""
    sources_loaded = []
    
    # The Shopping Cart Loop
    while True:
        print("=" * 70)
        print("📚 ADD DATA SOURCES (Build your knowledge base):")
        print("=" * 70)
        print("   1. Add Text File (.txt)")
        print("   2. Add PDF Document (.pdf)")
        print("   3. Add YouTube Video (URL)")
        print("   4. Add CSV File (.csv)")
        print("   5. DONE - Start Processing")
        print()
        
        # Show current status
        if sources_loaded:
            print(f"📊 Current Status: {len(sources_loaded)} source(s) loaded")
            print(f"📈 Total characters: {len(all_text):,}")
            print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        # Option 4: Done - break the loop
        if choice == "5":
            if not all_text:
                print("\n⚠️  No data loaded yet! Please add at least one source.")
                continue
            print("\n✅ Processing accumulated data...")
            break
        
        # Load data based on choice
        print()
        raw_text = None
        source_name = None
        
        if choice == "1":
            # Text File
            file_path = input("Enter text file path (or press Enter for 'data.txt'): ").strip()
            if not file_path:
                file_path = "data.txt"
            print(f"Loading: {file_path}")
            raw_text = reading_data(file_path)
            source_name = f"Text: {file_path}"
            
        elif choice == "2":
            # PDF Document
            pdf_path = input("Enter PDF file path: ").strip()
            print(f"Loading: {pdf_path}")
            raw_text = get_pdf_text(pdf_path)
            source_name = f"PDF: {pdf_path}"
            
        elif choice == "3":
            # YouTube Video
            video_url = input("Enter YouTube URL: ").strip()
            print(f"Loading: {video_url}")
            raw_text = get_youtube_transcript(video_url)
            source_name = f"YouTube: {video_url[:50]}..."

        elif choice == "4":
            # CSV File
            csv_path = input("Enter CSV file path: ").strip()
            print(f"Loading: {csv_path}")
            raw_text = get_csv_text(csv_path)
            source_name = f"CSV: {csv_path}"
            
        else:
            print("❌ Invalid choice! Please select 1-5.")
            continue
        
        # Safety Check: Validate loaded data
        if not raw_text or raw_text.startswith("❌"):
            print("\n⚠️  ERROR: Failed to load this source!")
            if raw_text:
                print(f"   {raw_text}")
            print("   Skipping this source. Try again or choose a different source.\n")
            continue
        
        # APPEND to accumulator (not overwrite!)
        all_text += "\n\n" + raw_text
        sources_loaded.append(source_name)
        
        # User Feedback
        print(f"\n✅ Added {source_name}")
        print(f"📊 Total length: {len(all_text):,} characters")
        print(f"📚 Sources loaded: {len(sources_loaded)}\n")
    
    # Show final summary
    print("\n" + "=" * 70)
    print("📦 KNOWLEDGE BASE SUMMARY")
    print("=" * 70)
    print(f"Total sources: {len(sources_loaded)}")
    print(f"Total characters: {len(all_text):,}")
    print(f"Estimated words: {len(all_text.split()):,}")
    print("\nSources:")
    for i, source in enumerate(sources_loaded, 1):
        print(f"  {i}. {source}")
    
    # Use all_text instead of raw_text
    raw_text = all_text
    
    # 2. Chunk the Data
    print("\n--- ✂️  CHUNKING DATA ---")
    config = CRAGConfig.from_env()
    text_chunks = get_chunks(raw_text, config.chunk_size, config.chunk_overlap)
    print(f"   ✅ Created {len(text_chunks)} chunks")
    
    # 3. Embed (Create Vector Embeddings)
    print("\n--- 🧠 CREATING EMBEDDINGS ---")
    print("   Loading embedding model...")
    db_vectors, model = vector_embedding(text_chunks)
    print(f"   ✅ Generated {len(db_vectors)} vectors")

    retriever = HybridRetriever(text_chunks, db_vectors, model, config)
    controller = CorrectiveRAGController(
        config,
        retriever,
        logger=get_logger("crag", config.log_level),
    )
    
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
        
        print("🔍 Evaluating retrieval quality...")
        try:
            response = controller.run(query)
            evaluation = response.evaluation
            print(
                "   📄 Retrieval quality: "
                f"{evaluation.decision} (max score {evaluation.max_score:.4f})"
            )
            print("🤖 Generating answer...")
            print(f"\n💬 Answer:\n{response.answer}")
        except Exception as e:
            print(f"\n⚠️  CRAG pipeline failed: {e}")
            print(f"\n💬 Answer:\n{NO_ANSWER_RESPONSE}")
        
        end_ts = time.time()
        print(f"\n⏱️  Time taken: {end_ts - start_ts:.2f}s")
        print("-" * 70)

if __name__ == "__main__":
    start_app()
