import json
from functools import cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import CHUNKS_PATH, INDEX_PATH, MODEL_NAME

TOP_K = 4


# Safe for CLI use: each invocation is a new process.
# Tests must call _load.cache_clear() between runs.
@cache
def _load() -> tuple[SentenceTransformer, faiss.Index, list[str]]:
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH) as f:
        chunks: list[str] = json.load(f)
    return model, index, chunks


def retrieve(query: str) -> list[str]:
    model, index, chunks = _load()
    embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embedding)
    k = min(TOP_K, len(chunks))
    _, indices = index.search(embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
