import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_query(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(f"- {c['text']}" for c in context_chunks)
    prompt = f"""
You are a helpful compliance assistant. Use the following context to answer the question:

Context:
{context}

Question: {query}
Answer:
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are empathetic and precise."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()
