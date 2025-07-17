import os
import json
import faiss
import numpy as np
from policy_pilot.embed_utils import embed_texts  # central embedding

# Paths
BASE_DIR    = os.getcwd()
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks",      "chunks.json")
INDEX_PATH  = os.path.join(BASE_DIR, "vector_store", "faiss.index")
ID_MAP_PATH = os.path.join(BASE_DIR, "vector_store", "id_map.json")


def load_chunks(limit: int = None) -> tuple[list[str], list[str]]:
    """
    Load chunk IDs and texts from disk.
    :param limit: if set, only return the first N chunks.
    """
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    if limit is not None:
        chunks = chunks[:limit]
    ids   = [c['id']   for c in chunks]
    texts = [c['text'] for c in chunks]
    return ids, texts


def build_faiss_index(limit: int = None, preview: int = 3) -> None:
    """
    1) Load up to `limit` chunks
    2) Embed them all at once via embed_utils.embed_texts()
    3) Preview the first `preview` vectors
    4) Persist embeddings (.npy), build & save FAISS index + ID map
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    # 1) Load
    ids, texts = load_chunks(limit)
    print(f"Loaded {len(ids)} chunks.")

    # 2) Embed
    embeddings = embed_texts(texts)

    # 3) Preview
    print(f"\nPreview of first {preview} embeddings (first 5 dims):")
    for cid, vec in zip(ids[:preview], embeddings[:preview]):
        print(f"  {cid}: {vec[:5]} ...")

    # 4a) Save raw embeddings
    emb_path = os.path.join(os.path.dirname(INDEX_PATH), "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"\nSaved embeddings to {emb_path}")

    # 4b) Build FAISS index
    arr = np.array(embeddings, dtype='float32')
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, INDEX_PATH)

    # 4c) Save ID map
    with open(ID_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(ids, f)

    print(f"Built FAISS index with {len(ids)} vectors.\n")


def query_faiss(query: str, top_k: int = 3) -> list[dict]:
    """
    Embed `query`, search the FAISS index, and return the top_k chunks.
    Each result is a dict with keys 'id', 'text', and 'score'.
    """
    # Load index + metadata
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAP_PATH, 'r', encoding='utf-8') as f:
        ids = json.load(f)
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunk_map = {c['id']: c['text'] for c in json.load(f)}

    # Embed and normalize query
    q_emb = embed_texts([query])
    q_arr = np.array(q_emb, dtype='float32')
    faiss.normalize_L2(q_arr)

    # Search
    distances, indices = index.search(q_arr, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        cid = ids[idx]
        results.append({
            "id":    cid,
            "text":  chunk_map[cid],
            "score": float(score)
        })
    return results



# import os
# import json
# import time
# import faiss
# import numpy as np
# from policy_pilot.embed_utils import embed_texts

# # File paths
# BASE_DIR       = os.getcwd()
# CHUNKS_PATH    = os.path.join(BASE_DIR, "chunks", "chunks.json")
# INDEX_PATH     = os.path.join(BASE_DIR, "vector_store", "faiss.index")
# ID_MAP_PATH    = os.path.join(BASE_DIR, "vector_store", "id_map.json")

# # Rate‑limit settings
# BATCH_SIZE     = 199     # Keep under 200 to respect 200 RPM limit
# SLEEP_INTERVAL = 65      # Seconds between batches


# def load_chunks(limit: int = None) -> tuple[list[str], list[str]]:
#     """
#     Load chunk IDs and texts from JSON file.

#     :param limit: Optional max number of chunks.
#     :return: Tuple of (ids, texts).
#     """
#     with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
#         chunks = json.load(f)
#     if limit:
#         chunks = chunks[:limit]
#     ids = [chunk['id'] for chunk in chunks]
#     texts = [chunk['text'] for chunk in chunks]
#     return ids, texts


# def embed_chunks(texts: list[str]) -> list[list[float]]:
#     """
#     Embed texts in rate‑limited batches.

#     :param texts: List of strings to embed.
#     :return: List of embedding vectors.
#     """
#     embeddings = []
#     total = len(texts)
#     for start in range(0, total, BATCH_SIZE):
#         end = min(start + BATCH_SIZE, total)
#         batch = texts[start:end]
#         print(f"Embedding batch {start+1}-{end} of {total}...")
#         t0 = time.time()
#         batch_embs = embed_texts(batch)
#         embeddings.extend(batch_embs)
#         elapsed = time.time() - t0
#         if end < total:
#             to_sleep = SLEEP_INTERVAL - elapsed
#             if to_sleep > 0:
#                 print(f"Sleeping for {to_sleep:.1f}s...")
#                 time.sleep(to_sleep)
#     return embeddings


# def build_faiss_index(limit: int = None) -> None:
#     """
#     Orchestrate chunk loading, embedding, and FAISS index creation.

#     :param limit: Optional maximum number of chunks to index.
#     """
#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

#     # Load and embed
#     ids, texts = load_chunks(limit)
#     print(f"Loaded {len(ids)} chunks; starting embedding...")
#     embs = embed_chunks(texts)

#     # Convert to NumPy and normalize
#     arr = np.array(embs, dtype='float32')
#     faiss.normalize_L2(arr)

#     # Build FAISS index
#     index = faiss.IndexFlatIP(arr.shape[1])
#     index.add(arr)

#     # Persist index and ID map
#     faiss.write_index(index, INDEX_PATH)
#     with open(ID_MAP_PATH, 'w', encoding='utf-8') as f:
#         json.dump(ids, f)

#     print(f"Built FAISS index with {len(ids)} vectors.")


# def query_faiss(query: str, top_k: int = 3) -> list[dict]:
#     """
#     Query the FAISS index for the top_k most similar chunks.

#     :param query: User query string.
#     :param top_k: Number of results to return.
#     :return: List of dicts with keys 'id', 'text', and 'score'.
#     """
#     # Load index and metadata
#     index = faiss.read_index(INDEX_PATH)
#     with open(ID_MAP_PATH, 'r', encoding='utf-8') as f:
#         ids = json.load(f)
#     with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
#         chunk_map = {c['id']: c['text'] for c in json.load(f)}

#     # Embed query and normalize
#     q_emb = np.array(embed_texts([query]), dtype='float32')
#     faiss.normalize_L2(q_emb)

#     # Search and collect results
#     distances, indices = index.search(q_emb, top_k)
#     results = []
#     for i, idx in enumerate(indices[0]):
#         cid = ids[idx]
#         results.append({
#             'id': cid,
#             'text': chunk_map[cid],
#             'score': float(distances[0][i])
#         })
#     return results