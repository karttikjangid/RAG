import torch
from sentence_transformers import util

def search_best_chunks(query, model, db_vectors, chunks, k=3):
    """
    Returns the TOP K chunks, not just the single best one.
    """
    # 1. Encode the query
    query_vector = model.encode(query)
    
    # 2. Calculate Cosine Similarity
    scores = util.cos_sim(query_vector, db_vectors)
    
    # 3. Get Top K Results
    # This gives us the top 3 scores and their index positions
    top_results = torch.topk(scores, k=k)
    
    # 4. Package the results
    results = []
    
    # Loop through the k results
    for i in range(k):
        # Get the index (location in the list)
        idx = top_results.indices[0][i].item()
        
        # Get the score (confidence level)
        score = top_results.values[0][i].item()
        
        # Add to our list
        results.append({
            "text": chunks[idx],
            "score": score
        })
        
    return results