"""
Real RAG Demo - Using actual RAG documentation PDF

This demonstrates the complete RAG pipeline using the 
'Retrieval Augmented Generation (RAG) for Everyone' PDF
"""

from pdf_ingestion import get_pdf_text
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import NO_ANSWER_RESPONSE, generate_answer

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120
MIN_SIMILARITY = 0.2  # Minimum cosine similarity (0.0-1.0) to trust retrieved context

def main():
    print("\n" + "=" * 70)
    print("📚 RAG SYSTEM - REAL DOCUMENT PROCESSING DEMO")
    print("=" * 70)
    
    # Step 1: Extract PDF text
    print("\n1️⃣  LOADING PDF DOCUMENT...")
    pdf_path = "../Retrieval Augmented Generation (RAG) for Everyone (1).pdf"
    
    text = get_pdf_text(pdf_path)
    
    if text.startswith("❌"):
        print(f"Error: {text}")
        return
    
    print(f"\n✅ Successfully loaded 57-page RAG document")
    print(f"   Characters: {len(text):,}")
    print(f"   Words: {len(text.split()):,}")
    
    # Step 2: Chunk the text
    print("\n2️⃣  CHUNKING TEXT...")
    chunks = get_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    print(
        f"   ✅ Created {len(chunks)} chunks "
        f"({DEFAULT_CHUNK_SIZE} chars each, {DEFAULT_CHUNK_OVERLAP} overlap)"
    )
    
    # Step 3: Create embeddings
    print("\n3️⃣  CREATING VECTOR EMBEDDINGS...")
    vectors, model = vector_embedding(chunks)
    print(f"   ✅ Generated {len(vectors)} vectors (384-dimensional)")
    
    # Step 4: Interactive Q&A
    print("\n" + "=" * 70)
    print("🤖 RAG SYSTEM READY - Ask questions about RAG!")
    print("=" * 70)
    
    # Predefined test questions
    test_questions = [
        "What is RAG?",
        "What are the components of RAG?",
        "What are the advantages of RAG?",
        "What is chunking in RAG?",
        "What is semantic caching?"
    ]
    
    print("\nTest Questions:")
    for i, q in enumerate(test_questions, 1):
        print(f"{i}. {q}")
    
    print("\nType a number (1-5) to ask a test question, or type your own question.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("❓ Your question: ").strip()
        
        if user_input.lower() == 'exit':
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Check if user selected a test question
        if user_input.isdigit() and 1 <= int(user_input) <= len(test_questions):
            query = test_questions[int(user_input) - 1]
            print(f"\nSelected: {query}")
        else:
            query = user_input
        
        print(f"\n🔍 Searching knowledge base...")
        
        # Retrieve relevant chunks
        results = search_best_chunks(query, model, vectors, chunks, k=3)
        
        best_result = results[0]
        print(f"\n📄 Best Match (confidence: {best_result['score']:.4f}):")
        print(f"   {best_result['text'][:200]}...")
        
        # Generate answer
        print(f"\n🤖 Generating answer...")
        try:
            if best_result["score"] < MIN_SIMILARITY:
                print(f"\n💬 Answer:\n{NO_ANSWER_RESPONSE}")
            else:
                answer = generate_answer(query, best_result["text"])
                print(f"\n💬 Answer:\n{answer}")
        except Exception as e:
            print(f"\n⚠️  LLM not available. Using retrieved context:")
            print(f"\n{best_result['text'][:500]}...")
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
