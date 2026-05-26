import pickle
from pathlib import Path

import faiss
import fitz  # pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("docuchat.index")
CHUNKS_PATH = Path("docuchat.chunks")
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def split_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def reset_index() -> None:
    INDEX_PATH.unlink(missing_ok=True)
    CHUNKS_PATH.unlink(missing_ok=True)


def build_index(pdf_paths: list[str]) -> None:
    all_chunks: list[str] = []
    for path in pdf_paths:
        print(f"Reading {path}...")
        text = load_pdf(path)
        chunks = split_text(text)
        all_chunks.extend(chunks)
        print(f"  {len(chunks)} chunks from {Path(path).name}")

    print(f"\nCreating embeddings for {len(all_chunks)} chunks...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        all_chunks, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(
        f"Index saved. {len(all_chunks)} chunks "
        f"from {len(pdf_paths)} document(s)."
    )
