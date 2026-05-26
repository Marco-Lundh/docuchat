import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import INDEX_PATH, CHUNKS_PATH, MODEL_NAME

TOP_K = 4

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[str] | None = None


def _load() -> None:
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None:
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)


def retrieve(query: str) -> list[str]:
    _load()
    embedding = _model.encode(
        [query], convert_to_numpy=True
    ).astype(np.float32)
    faiss.normalize_L2(embedding)
    k = min(TOP_K, len(_chunks))
    _, indices = _index.search(embedding, k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]
