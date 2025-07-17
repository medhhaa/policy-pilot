import os
import json
import faiss
import numpy as np
from policy_pilot.embed_utils import embed_texts

# Paths for FAISS index and ID map
INDEX_PATH = "vector_store/faiss.index"
ID_MAP_PATH = "vector_store/id_map.json"
# Path for chunks JSON
CHUNKS_PATH = "chunks/chunks.json"


def build_faiss_index(limit: int = None) -> None:
    '''
    Build a FAISS index from embedded chunks.
    Reads chunks from CHUNKS_PATH, computes embeddings, creates an IP index,
    normalizes vectors, and saves the index and ID map.

    :param limit: Optional max number of chunks to index (for testing).
    '''
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    # Load chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # Optionally limit the number of chunks for embedding
    if limit is not None:
        chunks = chunks[:limit]

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    # Embed texts
    embeddings = embed_texts(texts)
    arr = np.array(embeddings, dtype="float32")
    # Normalize for Inner Product similarity
    faiss.normalize_L2(arr)

    # Build index
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    # Save index and ID map
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"Built FAISS index with {len(ids)} vectors.")


def query_faiss(query: str, top_k: int = 3) -> list[dict]:
    '''
    Query the FAISS index for the top_k most similar chunks to the input query.

    :param query: The user's question or text to search.
    :param top_k: Number of nearest neighbors to return.
    :returns: List of dictionaries containing 'id' and 'text' of top chunks.
    '''
    # Load index and ID map
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)
    # Load chunk texts
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunk_map = {c["id"]: c["text"] for c in json.load(f)}

    # Embed query and normalize
    q_emb = np.array(embed_texts([query]), dtype="float32")
    faiss.normalize_L2(q_emb)

    # Search index
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        cid = ids[idx]
        results.append({"id": cid, "text": chunk_map[cid]})
    return results
