import requests
import json

def generate_answer(query, context):
    # 1. The Prompt Engineering (Crucial Step)
    # We explicitly tell the AI to ONLY use the context provided.
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the provided context. 
    If the answer is not in the context, say "I don't know."
    
    Context: {context}
    
    Question: {query}
    """
    
    # 2. Connect to Ollama
    # Ollama runs on port 11434 by default
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3.2", # OR "mistral" - whatever you pulled in terminal
        "prompt": prompt,
        "stream": False 
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Check for HTTP errors
        
        # 3. Parse the result
        return response.json()['response']
        
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"

# Test it independently first!
if __name__ == "__main__":
    test_context = "Badminton debuted in the Olympics in 1992. It is played with a shuttlecock."
    test_query = "When did badminton join the Olympics?"
    
    print("Asking Ollama...")
    answer = generate_answer(test_query, test_context)
    print("\nAnswer:", answer)