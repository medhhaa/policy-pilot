# embed_utils.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Convert a list of text strings to embeddings using OpenAI's new client API.
    
    Parameters:
        texts (list[str]): List of input text strings.
        model (str): OpenAI embedding model name.
    
    Returns:
        list[list[float]]: List of embeddings, one per input text.
    """
    # Use the embeddings.create method on the OpenAI client
    response = client.embeddings.create(model=model, input=texts)
    # Extract embedding vectors from response data
    return [data_point.embedding for data_point in response.data]
