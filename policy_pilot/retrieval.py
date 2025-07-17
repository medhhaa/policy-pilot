import os
import json
import faiss
import numpy as np
from policy_pilot.embed_utils import embed_texts  # central embedding

# Paths
BASE_DIR = os.getcwd()
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks", "chunks.json")
INDEX_PATH = os.path.join(BASE_DIR, "vector_store", "faiss.index")
ID_MAP_PATH = os.path.join(BASE_DIR, "vector_store", "id_map.json")



def load_chunks(limit: int = None) -> tuple[list[str], list[str]]:
    """
    Load chunk IDs and texts from disk.
    :param limit: if set, only return the first N chunks.
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if limit is not None:
        chunks = chunks[:limit]
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
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
    arr = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, INDEX_PATH)

    # 4c) Save ID map
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"Built FAISS index with {len(ids)} vectors.\n")


_index:   faiss.Index = None
_ids:     list[str]  = None
_chunk_map: dict[str,str] = None

# to optimize @query_faiss time
def _lazy_load_store():
    global _index, _ids, _chunk_map
    if _index is None:
        # 1) Load FAISS index
        _index = faiss.read_index(INDEX_PATH)
        # 2) Load ID list
        with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
            _ids = json.load(f)
        # 3) Load full chunk → text map
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        _chunk_map = {c["id"]: c["text"] for c in chunks}


def query_faiss(query: str, top_k: int = 3) -> list[dict]:
    """
    1) Lazily load index & metadata on first call
    2) Embed & normalize the query vector
    3) Perform k-NN search
    4) Map back to chunk IDs + texts
    """
    _lazy_load_store()

  # Embed & normalize
    q_emb = embed_texts([query])
    q_arr = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_arr)

    # Search
    distances, indices = _index.search(q_arr, top_k)

    # Collect results
    results = []
    for score, idx in zip(distances[0], indices[0]):
        cid = _ids[idx]
        results.append({
            "id":    cid,
            "text":  _chunk_map[cid],
            "score": float(score)
        })
    return results
