# Updated pdf_chunker.py with fallback to pdfminer for text extraction

"""
Extract and chunk text from compliance PDFs (GDPR, HIPAA, SOX, CCPA, PCI-DSS)
into manageable pieces (~750 characters each). 
First tries pdfplumber; if no text is extracted, falls back to pdfminer.six high-level extract_text.
Includes metadata for regulation and chunk index.
"""

import os
import json
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text

# Directory containing PDFs downloaded by pdf_fetcher.py
PDF_DIR = "pdfs"
# Output file for JSON chunks
CHUNKS_PATH = "chunks/chunks.json"

# Maximum characters per chunk
MAX_CHARS = 750

def chunk_text(text, max_len=MAX_CHARS):
    """
    Splits the text into chunks of ~max_len characters at sentence boundaries.
    """
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        segment = sentence.strip()
        if not segment:
            continue
        segment = segment + ("" if segment.endswith(".") else ".")
        if len(current) + len(segment) > max_len:
            chunks.append(current.strip())
            current = segment + " "
        else:
            current += segment + " "
    if current:
        chunks.append(current.strip())
    return chunks

def extract_and_chunk():
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    all_chunks = []

    # Iterate through each PDF in the pdfs/ directory
    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        reg_id = os.path.splitext(filename)[0]  # e.g., 'GDPR'
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"Processing {reg_id} ({pdf_path})")

        # Try pdfplumber first
        text_full = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_full += page_text + "\n"
        except Exception as e:
            print(f"  ⚠ pdfplumber failed for {filename}: {e}")

        # If no text extracted, fallback to pdfminer.six
        if not text_full.strip():
            try:
                text_full = pdfminer_extract_text(pdf_path)
                print(f"  ℹ Fallback to pdfminer for {filename}")
            except Exception as e:
                print(f"  ✖ Fallback pdfminer also failed for {filename}: {e}")
                continue  # skip this file entirely

        # Chunk the full text of the document
        chunks = chunk_text(text_full)
        for idx, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "id": f"{reg_id}_chunk{idx}",
                "regulation": reg_id,
                "text": chunk
            })

    # Save all chunks to JSON
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {CHUNKS_PATH}")

if __name__ == "__main__":
    extract_and_chunk()
