import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_index,
    load_pdf,
    reset_index,
    split_text,
)


def mock_sentence_transformer(
    mocker: MockerFixture, n_chunks: int = 1, dim: int = 384
) -> MagicMock:
    mock = mocker.MagicMock()
    mock.encode.return_value = np.random.rand(n_chunks, dim).astype(np.float32)
    mocker.patch("ingest.SentenceTransformer", return_value=mock)
    return mock


def test_load_pdf_extracts_text(sample_pdf: str) -> None:
    text = load_pdf(sample_pdf)
    assert "sample document" in text


def test_load_pdf_extracts_text_from_all_pages(multi_page_pdf: str) -> None:
    text = load_pdf(multi_page_pdf)
    assert "Page one" in text
    assert "Page two" in text


def test_load_pdf_returns_string(sample_pdf: str) -> None:
    assert isinstance(load_pdf(sample_pdf), str)


def test_split_text_empty_string_returns_empty_list() -> None:
    assert split_text("") == []


def test_split_text_whitespace_only_returns_empty_list() -> None:
    assert split_text("   \n\t  ") == []


def test_split_text_short_text_returns_single_chunk() -> None:
    text = " ".join(["word"] * 100)
    chunks = split_text(text)
    assert len(chunks) == 1


def test_split_text_long_text_returns_multiple_chunks() -> None:
    text = " ".join(["word"] * 1000)
    chunks = split_text(text)
    assert len(chunks) > 1


def test_split_text_chunk_does_not_exceed_max_size() -> None:
    text = " ".join(["word"] * 2000)
    for chunk in split_text(text):
        assert len(chunk.split()) <= CHUNK_SIZE


def test_split_text_consecutive_chunks_overlap() -> None:
    # With 500-word chunks and 50-word overlap, chunk N+1 starts
    # 450 words after chunk N — they share 50 words at the boundary.
    text = " ".join([str(i) for i in range(1000)])
    chunks = split_text(text)
    assert len(chunks) >= 2
    tail = chunks[0].split()[-CHUNK_OVERLAP:]
    head = chunks[1].split()[:CHUNK_OVERLAP]
    assert tail == head


def test_split_text_all_chunks_are_non_empty() -> None:
    text = " ".join(["word"] * 2000)
    for chunk in split_text(text):
        assert chunk.strip()


def test_reset_index_deletes_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_file = tmp_path / "docuchat.index"
    chunks_file = tmp_path / "docuchat.chunks"
    index_file.write_bytes(b"data")
    chunks_file.write_bytes(b"data")

    monkeypatch.setattr("ingest.INDEX_PATH", index_file)
    monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)

    reset_index()

    assert not index_file.exists()
    assert not chunks_file.exists()


def test_reset_index_tolerates_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ingest.INDEX_PATH", tmp_path / "missing.index")
    monkeypatch.setattr("ingest.CHUNKS_PATH", tmp_path / "missing.chunks")
    reset_index()


def test_build_index_creates_index_and_chunks_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    sample_pdf: str,
) -> None:
    index_file = tmp_path / "docuchat.index"
    chunks_file = tmp_path / "docuchat.chunks"
    monkeypatch.setattr("ingest.INDEX_PATH", index_file)
    monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)
    mock_sentence_transformer(mocker)

    build_index([sample_pdf])

    assert index_file.exists()
    assert chunks_file.exists()


def test_build_index_chunks_file_contains_text_from_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    sample_pdf: str,
) -> None:
    chunks_file = tmp_path / "docuchat.chunks"
    monkeypatch.setattr("ingest.INDEX_PATH", tmp_path / "docuchat.index")
    monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)
    mock_sentence_transformer(mocker)

    build_index([sample_pdf])

    with open(chunks_file) as f:
        chunks = json.load(f)
    assert any("sample document" in c for c in chunks)


def test_build_index_combines_chunks_from_multiple_pdfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    sample_pdf: str,
    another_pdf: str,
) -> None:
    chunks_file = tmp_path / "docuchat.chunks"
    monkeypatch.setattr("ingest.INDEX_PATH", tmp_path / "docuchat.index")
    monkeypatch.setattr("ingest.CHUNKS_PATH", chunks_file)

    mock = mocker.MagicMock()

    def fake_encode(texts: list[str], **_: object) -> np.ndarray:
        return np.random.rand(len(texts), 384).astype(np.float32)

    mock.encode.side_effect = fake_encode
    mocker.patch("ingest.SentenceTransformer", return_value=mock)

    build_index([sample_pdf, another_pdf])

    with open(chunks_file) as f:
        chunks = json.load(f)
    texts = " ".join(chunks)
    assert "sample document" in texts
    assert "unique content" in texts
