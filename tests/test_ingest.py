import pickle

import numpy as np

from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_index,
    load_pdf,
    reset_index,
    split_text,
)


class TestLoadPdf:
    def test_extracts_text(self, sample_pdf):
        text = load_pdf(sample_pdf)
        assert "sample document" in text

    def test_extracts_text_from_all_pages(self, multi_page_pdf):
        text = load_pdf(multi_page_pdf)
        assert "Page one" in text
        assert "Page two" in text

    def test_returns_string(self, sample_pdf):
        assert isinstance(load_pdf(sample_pdf), str)


class TestSplitText:
    def test_empty_string_returns_empty_list(self):
        assert split_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_text("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        text = " ".join(["word"] * 100)
        chunks = split_text(text)
        assert len(chunks) == 1

    def test_long_text_returns_multiple_chunks(self):
        text = " ".join(["word"] * 1000)
        chunks = split_text(text)
        assert len(chunks) > 1

    def test_chunk_does_not_exceed_max_size(self):
        text = " ".join(["word"] * 2000)
        for chunk in split_text(text):
            assert len(chunk.split()) <= CHUNK_SIZE

    def test_consecutive_chunks_overlap(self):
        # With 500-word chunks and 50-word overlap, chunk N+1 starts
        # 450 words after chunk N — they share 50 words at the boundary.
        text = " ".join([str(i) for i in range(1000)])
        chunks = split_text(text)
        assert len(chunks) >= 2
        tail = chunks[0].split()[-CHUNK_OVERLAP:]
        head = chunks[1].split()[:CHUNK_OVERLAP]
        assert tail == head

    def test_all_chunks_are_non_empty(self):
        text = " ".join(["word"] * 2000)
        for chunk in split_text(text):
            assert chunk.strip()


class TestResetIndex:
    def test_deletes_both_files(self, tmp_path, monkeypatch):
        index_file = tmp_path / "docuchat.index"
        chunks_file = tmp_path / "docuchat.chunks"
        index_file.write_bytes(b"data")
        chunks_file.write_bytes(b"data")

        monkeypatch.setattr("ingest.INDEX_PATH", index_file)
        monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)

        reset_index()

        assert not index_file.exists()
        assert not chunks_file.exists()

    def test_tolerates_missing_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingest.INDEX_PATH", tmp_path / "missing.index")
        monkeypatch.setattr("ingest.CHUNKS_PATH", tmp_path / "missing.chunks")
        reset_index()  # must not raise


class TestBuildIndex:
    def _mock_model(self, mocker, n_chunks=1, dim=384):
        mock = mocker.MagicMock()
        mock.encode.return_value = (
            np.random.rand(n_chunks, dim).astype(np.float32)
        )
        mocker.patch("ingest.SentenceTransformer", return_value=mock)
        return mock

    def test_creates_index_and_chunks_files(
        self, tmp_path, monkeypatch, mocker, sample_pdf
    ):
        index_file = tmp_path / "docuchat.index"
        chunks_file = tmp_path / "docuchat.chunks"
        monkeypatch.setattr("ingest.INDEX_PATH", index_file)
        monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)
        self._mock_model(mocker)

        build_index([sample_pdf])

        assert index_file.exists()
        assert chunks_file.exists()

    def test_chunks_file_contains_text_from_pdf(
        self, tmp_path, monkeypatch, mocker, sample_pdf
    ):
        chunks_file = tmp_path / "docuchat.chunks"
        monkeypatch.setattr("ingest.INDEX_PATH", tmp_path / "docuchat.index")
        monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)
        self._mock_model(mocker)

        build_index([sample_pdf])

        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        assert any("sample document" in c for c in chunks)

    def test_combines_chunks_from_multiple_pdfs(
        self, tmp_path, monkeypatch, mocker, sample_pdf, another_pdf
    ):
        chunks_file = tmp_path / "docuchat.chunks"
        monkeypatch.setattr("ingest.INDEX_PATH", tmp_path / "docuchat.index")
        monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)

        mock = mocker.MagicMock()
        call_count = 0

        def fake_encode(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            return np.random.rand(len(texts), 384).astype(np.float32)

        mock.encode.side_effect = fake_encode
        mocker.patch("ingest.SentenceTransformer", return_value=mock)

        build_index([sample_pdf, another_pdf])

        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        texts = " ".join(chunks)
        assert "sample document" in texts
        assert "unique content" in texts
