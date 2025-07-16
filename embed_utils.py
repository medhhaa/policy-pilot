import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    '''
    Convert a list of text strings to embeddings using OpenAI's embedding API.
    
    Parameters:
        texts (list[str]): List of input text strings.
        model (str): OpenAI embedding model name.
    
    Returns:
        list[list[float]]: List of embeddings, one per input text.
    '''
    response = openai.Embedding.create(model=model, input=texts)
    return [item["embedding"] for item in response["data"]]