"""
Real RAG Demo - Using actual RAG documentation PDF

This demonstrates the complete RAG pipeline using the 
'Retrieval Augmented Generation (RAG) for Everyone' PDF
"""

from pdf_ingestion import get_pdf_text
from chunking import get_chunks
from embedding import vector_embedding
from generation import NO_ANSWER_RESPONSE
from crag.config import CRAGConfig
from crag.controller import CorrectiveRAGController
from crag.hybrid_retrieval import HybridRetriever
from crag.logging_utils import get_logger


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
    config = CRAGConfig.from_env()
    chunks = get_chunks(text, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    print(
        f"   ✅ Created {len(chunks)} chunks "
        f"({config.chunk_size} chars each, {config.chunk_overlap} overlap)"
    )
    
    # Step 3: Create embeddings
    print("\n3️⃣  CREATING VECTOR EMBEDDINGS...")
    vectors, model = vector_embedding(chunks)
    print(f"   ✅ Generated {len(vectors)} vectors (384-dimensional)")

    retriever = HybridRetriever(chunks, vectors, model, config)
    controller = CorrectiveRAGController(
        config,
        retriever,
        logger=get_logger("crag", config.log_level),
    )
    
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
        
        print(f"\n🤖 Running Corrective RAG...")
        try:
            response = controller.run(query)
            evaluation = response.evaluation
            print(
                "\n📄 Retrieval quality: "
                f"{evaluation.decision} (max score {evaluation.max_score:.4f})"
            )
            print(f"\n💬 Answer:\n{response.answer}")
        except Exception as e:
            print(f"\n⚠️  CRAG pipeline failed: {e}")
            print(f"\n💬 Answer:\n{NO_ANSWER_RESPONSE}")
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
