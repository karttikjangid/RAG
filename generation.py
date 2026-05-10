import os
import requests

NO_ANSWER_RESPONSE = (
    "The uploaded document does not contain enough information to answer this question."
)

GEMINI_MODEL = "gemini-1.5-flash"


def _build_prompt(query, context):
    return (
        "You are a grounded assistant for a RAG system. "
        "Use ONLY the provided context to answer. "
        f'If the answer is not present, respond with exactly: "{NO_ANSWER_RESPONSE}".\n\n'
        f"Context:\n{context}\n\nQuestion:\n{query}\n"
    )


def generate_answer(query, context):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            'GEMINI_API_KEY environment variable is not set. '
            'Please set it using: export GEMINI_API_KEY="your_api_key"'
        )

    prompt = _build_prompt(query, context)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "topP": 0.9,
        },
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return NO_ANSWER_RESPONSE

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return NO_ANSWER_RESPONSE

    text = parts[0].get("text", "").strip()
    return text or NO_ANSWER_RESPONSE
