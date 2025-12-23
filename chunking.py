from data_ingestion import reading_data
sourced_text  = reading_data("data.txt")


def get_chunks(text , chunk_size , overlap):
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
    chunked = get_chunks(sourced_text , 200 , 50)
    print(f"Total chunks: {len(chunked)}")
    for i, chunk in enumerate(chunked[:3]):  # Show first 3
        print(f"\nChunk {i+1}:\n{chunk}")


    

