# retrieval.py
import torch
from sentence_transformers import util

def search_best_chunk(query, model, db_vectors, chunks):
    """
    Takes the query, model, vectors, and text chunks.
    Does the math to find the best match.
    Returns: (Best Text, Score)
    """
    
    # 1. Encode the Query (using the model passed from main.py)
    query_vector = model.encode(query)
    
    # 2. Do the Math (Cosine Similarity)
    scores = util.cos_sim(query_vector, db_vectors)
    
    # 3. Find the Winner
    best_index = torch.argmax(scores)
    
    # 4. Return the text and the score
    return chunks[best_index], scores[0][best_index]