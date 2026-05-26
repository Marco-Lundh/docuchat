import fitz
import pytest

from create_test_pdf import create_pdf


def make_pdf(path, pages: list[str]) -> str:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), text, fontsize=11)
    doc.save(str(path))
    return str(path)


@pytest.fixture
def sample_pdf(tmp_path):
    """Single-page PDF with a short paragraph."""
    return make_pdf(
        tmp_path / "sample.pdf",
        ["This is a sample document. It has some text for testing."],
    )


@pytest.fixture
def multi_page_pdf(tmp_path):
    """Two-page PDF."""
    return make_pdf(
        tmp_path / "multipage.pdf",
        ["Page one content.", "Page two content."],
    )


@pytest.fixture
def large_pdf(tmp_path):
    """PDF with enough words to produce multiple chunks (>500 words)."""
    words = " ".join([f"word{i}" for i in range(600)])
    return make_pdf(tmp_path / "large.pdf", [words])


@pytest.fixture
def another_pdf(tmp_path):
    """A second distinct PDF for multi-document tests."""
    return make_pdf(
        tmp_path / "another.pdf",
        ["Completely different document with unique content."],
    )


@pytest.fixture
def handbook_pdf(tmp_path):
    """Realistic multi-section employee handbook PDF (Granit Software AB)."""
    return create_pdf(str(tmp_path / "handbook.pdf"))
