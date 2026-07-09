"""
build_index.py
Chunks the processed text documents, embeds each chunk using a local
sentence-transformers model, and builds a FAISS index for retrieval.

Usage:
    python src/build_index.py
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this scale
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks, to preserve context across boundaries


def chunk_documents(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split a document's text into overlapping chunks.
    Overlap ensures a sentence/rule split across a chunk boundary isn't lost entirely.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # step forward, but overlap with previous chunk
    return chunks


def embed_text(model: SentenceTransformer, texts):
    """Embed a list of strings into dense vectors (normalized for cosine similarity via inner product)."""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    return embeddings


def build_index():
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found at {manifest_path}. Run ingest.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []       # list of chunk text
    all_metadata = []      # list of dicts: {source_file, source_url, chunk_index}

    for txt_filename, info in manifest.items():
        filepath = os.path.join(PROCESSED_DIR, txt_filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_documents(text)
        print(f"  {txt_filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source_file": txt_filename,
                "source_url": info.get("source_url", "unknown"),
                "chunk_index": i,
            })

    if not all_chunks:
        print("[!] No chunks produced. Check that data/processed/ contains extracted text files.")
        return

    print(f"\nEmbedding {len(all_chunks)} chunks ...")
    embeddings = embed_text(model, all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "metadata": all_metadata}, f, indent=2)

    print(f"\nDone. FAISS index with {index.ntotal} vectors saved to {INDEX_DIR}/faiss.index")


if __name__ == "__main__":
    build_index()
