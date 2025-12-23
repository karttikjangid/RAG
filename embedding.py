from sentence_transformers import SentenceTransformer
from chunking import get_chunks
from data_ingestion import reading_data
sourced_text  = reading_data("data.txt")

chunked = get_chunks(sourced_text , 150 , 50)
def vector_embedding(chunked):
    print("loading embedding model.......")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunked)
    print(f"Created {len(vectors)} vectors.")
    print(f"Shape of first vector: {vectors[0].shape}")
    return vectors , model 


if __name__ == "__main__":
    vector_embedding(chunked)