# embed_rate_limited.py

"""
Rate-limited embedding script to batch embeddings without exceeding 
OpenAI's rate limits (200 requests per minute). 
Uses a buffer: batch_size=199 and interval_sec=65 seconds.
"""

import time
import json
from policy_pilot.embed_utils import embed_texts

def rate_limited_embeddings(texts, batch_size=199, interval_sec=65):
    """
    Embed texts in batches while respecting OpenAI rate limits.
    
    :param texts: List of text strings to embed.
    :param batch_size: Max number of texts per API call.
    :param interval_sec: Seconds to wait between consecutive batches.
    :returns: List of embedding vectors.
    """
    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        
        print(f"Embedding batch {start+1}-{end} of {total} texts...")
        t0 = time.time()
        # Single API call for this batch
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        
        elapsed = time.time() - t0
        # Only sleep if there are more batches left
        if end < total:
            sleep_time = interval_sec - elapsed
            if sleep_time > 0:
                print(f"Sleeping {sleep_time:.1f}s to respect rate limits...")
                time.sleep(sleep_time)
    return all_embeddings

if __name__ == "__main__":
    # Load chunk texts
    chunks = json.load(open("chunks/chunks.json", "r", encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    
    # Generate embeddings with rate limiting
    embeddings = rate_limited_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings.")
