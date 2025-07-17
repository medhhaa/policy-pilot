import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load your Gemini/Vertex AI API key from .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Instantiate the GenAI client
client = genai.Client(api_key=API_KEY)

# The exact model alias 
MODEL_NAME = "gemini-2.5-pro"

# A strong system prompt to lock in persona and scope
SYSTEM_PROMPT = """
You are a senior regulatory compliance advisor with 15+ years’ experience working alongside government agencies. 
You provide clear, concise, and legally accurate guidance, and you never invent facts or cite anything outside the supplied context.
"""


def answer_query(query: str, context_chunks: list[dict]) -> str:
    # 1) Build a single-context string
    context = "\n\n".join(f"- {c['text']}" for c in context_chunks)

    # 2) User instructions: scope & format
    USER_PROMPT = f"""\
Context:
[{context}]

Question:
{query}

Instructions:
1. Use only the information in the Context.
2. If the provided context doesn’t cover your question, summarize what you do know from the context, then answer from general compliance best practices.
3. Structure your answer with a brief summary, then bullet‑pointed recommendations.
4. Keep language precise and actionable; do not speculate or add external details.
"""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        candidate_count=1,
      
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=USER_PROMPT,
        config=config
    )                                                                      

    return response.text
