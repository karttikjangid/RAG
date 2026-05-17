DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120

def get_chunks(text, chunk_size, overlap):
    sliced = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks = text[start:end]
        sliced.append(chunks)
        start  += (chunk_size - overlap)

    return sliced


if __name__ == "__main__":
    from data_ingestion import reading_data

    sourced_text = reading_data("data.txt")
    chunked = get_chunks(sourced_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    print(f"Total chunks: {len(chunked)}")
    for i, chunk in enumerate(chunked[:3]):
        print(f"\nChunk {i+1}:\n{chunk}")


    
