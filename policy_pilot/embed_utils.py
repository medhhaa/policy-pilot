# embed_utils.py

from sentence_transformers import SentenceTransformer

# Load the MiniLM model once at import time
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using a local SentenceTransformer model.
    No API calls or rate limits.
    """
    # encode returns a numpy array of shape (len(texts), dim)
    embeddings = _model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()
