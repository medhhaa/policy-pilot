import os
from dotenv import load_dotenv
import google.genai as genai

# Load .env and configure the Gemini Pro client
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# The exact model alias you want to call
MODEL_NAME = "gemini-pro-preview"

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
2. If the Context doesn’t contain enough to answer, reply:
   “Insufficient information provided to answer the question.”
3. Structure your answer with a brief summary, then bullet‑pointed recommendations.
4. Keep language precise and actionable; do not speculate or add external details.
"""

    # 3) Call Gemini Pro
    resp = genai.chat.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"author": "system", "content": SYSTEM_PROMPT},
            {"author": "user", "content": USER_PROMPT},
        ],
    )

    # 4) Extract and return the answer text
    return resp.candidates[0].content.strip()
