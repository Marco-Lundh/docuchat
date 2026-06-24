import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
from pytest_mock import MockerFixture

import retriever


@pytest.fixture(autouse=True)
def clear_load_cache() -> Generator[None, None, None]:
    retriever._load.cache_clear()
    yield
    retriever._load.cache_clear()


def make_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    v = vectors.copy().astype(np.float32)
    faiss.normalize_L2(v)
    index = faiss.IndexFlatIP(dim)
    index.add(v)
    return index


def inject_retriever(
    mocker: MockerFixture,
    chunks: list[str],
    query_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, MagicMock]:
    dim = 8
    n = len(chunks)
    doc_vecs = np.random.rand(n, dim).astype(np.float32)
    index = make_index(doc_vecs)
    if query_vector is None:
        query_vector = np.random.rand(1, dim).astype(np.float32)
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = query_vector
    mocker.patch(
        "retriever._load",
        return_value=(mock_model, index, chunks),
    )
    return doc_vecs, query_vector, mock_model


def test_make_index_returns_faiss_index() -> None:
    vecs = np.random.rand(3, 8).astype(np.float32)
    index = make_index(vecs)
    assert isinstance(index, faiss.Index)


def test_make_index_contains_all_vectors() -> None:
    vecs = np.random.rand(5, 8).astype(np.float32)
    index = make_index(vecs)
    assert index.ntotal == 5


def test_make_index_does_not_mutate_input() -> None:
    vecs = np.random.rand(3, 8).astype(np.float32)
    original = vecs.copy()
    make_index(vecs)
    np.testing.assert_array_equal(vecs, original)


def test_retrieve_returns_list_of_strings(mocker: MockerFixture) -> None:
    inject_retriever(mocker, ["chunk a", "chunk b", "chunk c"])
    result = retriever.retrieve("some query")
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


def test_retrieve_returns_at_most_top_k(mocker: MockerFixture) -> None:
    chunks = [f"chunk {i}" for i in range(10)]
    inject_retriever(mocker, chunks)
    result = retriever.retrieve("query")
    assert len(result) <= retriever.TOP_K


def test_retrieve_returns_all_when_fewer_than_top_k(
    mocker: MockerFixture,
) -> None:
    inject_retriever(mocker, ["only one chunk"])
    result = retriever.retrieve("query")
    assert len(result) == 1


def test_retrieve_returns_most_similar_chunk(mocker: MockerFixture) -> None:
    dim = 8
    query = np.ones((1, dim), dtype=np.float32)
    doc_vecs = np.array(
        [
            np.ones(dim),
            -np.ones(dim),
        ],
        dtype=np.float32,
    )
    chunks = ["similar chunk", "opposite chunk"]
    index = make_index(doc_vecs)
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = query
    mocker.patch(
        "retriever._load",
        return_value=(mock_model, index, chunks),
    )
    result = retriever.retrieve("anything")
    assert result[0] == "similar chunk"


def test_retrieve_encodes_the_query(mocker: MockerFixture) -> None:
    _, _, mock_model = inject_retriever(mocker, ["chunk"])
    retriever.retrieve("my question")
    mock_model.encode.assert_called_once_with(
        ["my question"], convert_to_numpy=True
    )


def test_load_returns_model_index_and_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    dim = 8
    vecs = np.random.rand(3, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    index_path = tmp_path / "test.index"
    chunks_path = tmp_path / "test.chunks"
    faiss.write_index(idx, str(index_path))
    with open(chunks_path, "w") as f:
        json.dump(["a", "b", "c"], f)

    monkeypatch.setattr("retriever.INDEX_PATH", index_path)
    monkeypatch.setattr("retriever.CHUNKS_PATH", chunks_path)
    mock_model = mocker.MagicMock()
    mocker.patch("retriever.SentenceTransformer", return_value=mock_model)

    model, loaded_idx, chunks = retriever._load()

    assert model is mock_model
    assert loaded_idx.ntotal == 3
    assert chunks == ["a", "b", "c"]
