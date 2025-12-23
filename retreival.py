from sentence_transformers import SentenceTransformer, util
from chunking import get_chunks
from embedding import vector_embedding
from data_ingestion import reading_data
import torch

# 1. Load Data & Model (Do this ONCE, outside the loop)
sourced_text = reading_data("data.txt")
chunked = get_chunks(sourced_text, 150, 50)
db_vectors, model = vector_embedding(chunked) # We get the model here!

def search_best_chunk(query):
    # 2. Encode Query using the ALREADY LOADED model
    # We do not call vector_embedding() again because that would be slow.
    query_vector = model.encode(query)
    
    # 3. Calculate Similarity
    # util.cos_sim automatically converts numpy arrays to tensors, so we don't need manual torch.tensor() calls usually.
    scores = util.cos_sim(query_vector, db_vectors)
    
    # 4. Find Best Match
    best_index = torch.argmax(scores)
    
    # Return the text and the score
    return chunked[best_index], scores[0][best_index]

if __name__ == "__main__":
    query = "When was badminton officially included in the olympics?"
    
    result, score = search_best_chunk(query)
    
    print(f"Query: {query}")
    print(f"Match Score: {score:.4f}") 
    print("------------------------------------------------")
    print(f"Retrieved Context: {result}")