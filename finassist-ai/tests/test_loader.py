"""Unit tests for backend.rag.loader."""

import pytest

from backend.rag.loader import DocumentLoader, UnsupportedFileTypeError


def test_load_txt_cleans_whitespace(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Hello   world.\n\n\n\nThis  has   extra   spaces.")

    loader = DocumentLoader()
    doc = loader.load(path)

    assert "   " not in doc.text
    assert "\n\n\n" not in doc.text
    assert doc.filename == "doc.txt"


def test_load_markdown_detects_headings(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Revenue\n\nSome content here.\n\n## Margins\n\nMore content.")

    loader = DocumentLoader()
    doc = loader.load(path)

    assert "Revenue" in doc.headings
    assert "Margins" in doc.headings


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "doc.xyz"
    path.write_text("data")

    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError):
        loader.load(path)


def test_missing_file_raises(tmp_path):
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "does_not_exist.txt")
