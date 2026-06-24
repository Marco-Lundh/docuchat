from pathlib import Path

import fitz
import pytest


def make_pdf(path: Path | str, pages: list[str]) -> str:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), text, fontsize=11)
    doc.save(str(path))
    return str(path)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> str:
    """Single-page PDF with a short paragraph."""
    return make_pdf(
        tmp_path / "sample.pdf",
        ["This is a sample document. It has some text for testing."],
    )


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> str:
    """Two-page PDF."""
    return make_pdf(
        tmp_path / "multipage.pdf",
        ["Page one content.", "Page two content."],
    )


@pytest.fixture
def another_pdf(tmp_path: Path) -> str:
    """A second distinct PDF for multi-document tests."""
    return make_pdf(
        tmp_path / "another.pdf",
        ["Completely different document with unique content."],
    )
