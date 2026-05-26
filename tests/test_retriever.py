import numpy as np
import faiss
import pytest

import retriever


@pytest.fixture(autouse=True)
def reset_retriever_globals():
    """Clear cached model/index/chunks between tests."""
    retriever._model = None
    retriever._index = None
    retriever._chunks = None
    yield
    retriever._model = None
    retriever._index = None
    retriever._chunks = None


def make_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    v = vectors.copy().astype(np.float32)
    faiss.normalize_L2(v)
    index = faiss.IndexFlatIP(dim)
    index.add(v)
    return index


class TestRetrieve:
    def _inject(self, mocker, chunks: list[str], query_vector=None):
        """Inject a tiny real FAISS index and mock the encoder."""
        dim = 8
        n = len(chunks)
        doc_vecs = np.random.rand(n, dim).astype(np.float32)

        retriever._chunks = chunks
        retriever._index = make_index(doc_vecs)

        if query_vector is None:
            query_vector = np.random.rand(1, dim).astype(np.float32)

        mock_model = mocker.MagicMock()
        mock_model.encode.return_value = query_vector
        retriever._model = mock_model

        mocker.patch("retriever._load")  # prevent real loading
        return doc_vecs, query_vector

    def test_returns_list_of_strings(self, mocker):
        self._inject(mocker, ["chunk a", "chunk b", "chunk c"])
        result = retriever.retrieve("some query")
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_returns_at_most_top_k(self, mocker):
        chunks = [f"chunk {i}" for i in range(10)]
        self._inject(mocker, chunks)
        result = retriever.retrieve("query")
        assert len(result) <= retriever.TOP_K

    def test_returns_all_when_fewer_than_top_k(self, mocker):
        chunks = ["only one chunk"]
        self._inject(mocker, chunks)
        result = retriever.retrieve("query")
        assert len(result) == 1

    def test_returns_most_similar_chunk(self, mocker):
        dim = 8
        # Create a query vector and a near-identical document vector
        query = np.ones((1, dim), dtype=np.float32)
        doc_vecs = np.array(
            [
                np.ones(dim),        # very similar to query
                -np.ones(dim),       # opposite direction
            ],
            dtype=np.float32,
        )

        retriever._chunks = ["similar chunk", "opposite chunk"]
        retriever._index = make_index(doc_vecs)
        mock_model = mocker.MagicMock()
        mock_model.encode.return_value = query
        retriever._model = mock_model
        mocker.patch("retriever._load")

        result = retriever.retrieve("anything")
        assert result[0] == "similar chunk"

    def test_encodes_the_query(self, mocker):
        self._inject(mocker, ["chunk"])
        retriever.retrieve("my question")
        retriever._model.encode.assert_called_once_with(
            ["my question"], convert_to_numpy=True
        )
