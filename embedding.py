from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def vector_embedding(chunks, model=None):
    print("loading embedding model.......")
    model = model or load_embedding_model()
    vectors = model.encode(chunks)
    print(f"Created {len(vectors)} vectors.")
    print(f"Shape of first vector: {vectors[0].shape}")
    return vectors, model


if __name__ == "__main__":
    from chunking import get_chunks
    from data_ingestion import reading_data

    sourced_text = reading_data("data.txt")
    chunked = get_chunks(sourced_text, 150, 50)
    vector_embedding(chunked)
