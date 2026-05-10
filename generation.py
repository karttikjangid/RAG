import os
from openai import OpenAI
from google import genai

NO_ANSWER_RESPONSE = (
    "The uploaded document does not contain enough information to answer this question."
)

GEMINI_MODEL = "gemini-3-flash-preview"
NVIDIA_MODEL = "deepseek-ai/deepseek-v4-flash"

def _build_prompt(query, context):
    return (
        "You are a grounded assistant for a RAG system. "
        "Use ONLY the provided context to answer. "
        f'If the answer is not present, respond with exactly: "{NO_ANSWER_RESPONSE}".\n\n'
        f"Context:\n{context}\n\nQuestion:\n{query}\n"
    )

def generate_answer(query, context):
    prompt = _build_prompt(query, context)
    
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if nvidia_api_key:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        try:
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
                top_p=0.9,
            )
            content = response.choices[0].message.content
            return content.strip() if content else NO_ANSWER_RESPONSE
        except Exception as e:
            raise RuntimeError(f"Failed to connect to NVIDIA API: {e}")
            
    elif gemini_api_key:
        try:
            client = genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL, 
                contents=prompt
            )
            return response.text.strip() if response.text else NO_ANSWER_RESPONSE
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gemini API: {e}")
            
    else:
        raise ValueError(
            'Neither NVIDIA_API_KEY nor GEMINI_API_KEY environment variables are set. '
            'Please set at least one to use the generation feature.'
        )
