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

# Enhanced system prompt for better compliance guidance
SYSTEM_PROMPT = """
You are a senior regulatory compliance advisor with 15+ years of experience working with government agencies and Fortune 500 companies. 

Your role is to provide clear, actionable compliance guidance based on official regulatory documents. You must:

1. ALWAYS cite specific regulation sections (e.g., "HIPAA §164.312(a)(2)(iv)", "GDPR Article 32") when making recommendations
2. Structure responses with clear headings and bullet points for easy scanning
3. Distinguish between REQUIRED, RECOMMENDED, and OPTIONAL measures
4. Provide specific implementation steps when possible
5. Never invent facts or cite regulations not in the provided context
6. Use professional but accessible language suitable for both legal and technical teams

Response Format:
- Start with a direct answer to the question
- Use bullet points for requirements and recommendations
- Include specific regulation references in parentheses
- End with practical next steps when appropriate
"""


def answer_query(query: str, context_chunks: list[dict]) -> dict:
    """
    Enhanced function that returns structured response with metadata
    """
    # 1) Build context with better formatting
    context_sections = []
    source_regulations = set()
    
    for chunk in context_chunks:
        regulation = chunk.get('id', '').split('_')[0]
        source_regulations.add(regulation)
        context_sections.append(f"[{regulation}] {chunk['text']}")
    
    context = "\n\n".join(context_sections)

    # 2) Enhanced user prompt with specific formatting requirements
    USER_PROMPT = f"""
Context from official regulatory documents:
{context}

User Question: {query}

Please provide a structured compliance response following these guidelines:

1. Lead with a direct answer addressing the specific question
2. Use bullet points for requirements, with regulation citations in parentheses
3. Clearly distinguish between:
   • REQUIRED: Legal obligations that must be met
   • RECOMMENDED: Best practices to exceed minimum requirements

4. Include specific implementation guidance where possible
5. End with actionable next steps

Format your response with clear headings and professional language suitable for compliance teams.
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

    # Return structured response with metadata
    return {
        "answer": response.text,
        "sources": list(source_regulations),
        "context_chunks": len(context_chunks),
        "confidence": _calculate_confidence(context_chunks)
    }


def _calculate_confidence(context_chunks: list[dict]) -> str:
    """
    Calculate confidence level based on context quality
    """
    if not context_chunks:
        return "Low"
    
    # Simple confidence calculation based on similarity scores
    avg_score = sum(chunk.get('score', 0) for chunk in context_chunks) / len(context_chunks)
    
    if avg_score > 0.8:
        return "High"
    elif avg_score > 0.6:
        return "Medium"
    else:
        return "Low"