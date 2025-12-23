# main.py
import time
from data_ingestion import reading_data
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunk
from generation import generate_answer

def start_app():
    print("\n--- 🚀 INITIALIZING RAG SYSTEM ---")
    
    # 1. Load & Chunk (The Raw Material)
    print("Step 1: Loading Data...")
    raw_text = reading_data("data.txt")
    text_chunks = get_chunks(raw_text, 150, 50)
    print(f"   -> Created {len(text_chunks)} chunks.")
    
    # 2. Embed (The Brain)
    print("Step 2: Embedding Data (Loading Model)...")
    # We capture the 'vectors' and 'model' here!
    db_vectors, model = vector_embedding(text_chunks)
    print("   -> Embeddings ready.")
    
    print("\n--- ✅ SYSTEM READY (Type 'exit' to quit) ---")

    # 3. The Chat Loop
    while True:
        query = input("\nUser: ")
        
        if query.lower() == "exit":
            print("Bye!")
            break
        if not query.strip():
            continue
            
        start_ts = time.time()
        
        # A. RETRIEVAL
        # We pass the loaded model & vectors here!
        best_context, score = search_best_chunk(query, model, db_vectors, text_chunks)
        
        print(f"\n🔍 Context Found (Confidence: {score:.2f}):")
        print(f"\"{best_context[:100]}...\"") 
        
        # B. GENERATION
        print("🤖 Thinking...")
        answer = generate_answer(query, best_context)
        
        end_ts = time.time()
        
        print(f"\nAI Answer:\n{answer}")
        print(f"\n(Time taken: {end_ts - start_ts:.2f}s)")

if __name__ == "__main__":
    start_app()