"""Shared pytest fixtures for FinAssist AI tests.

Tests avoid network calls and real model downloads by using a temp
working directory for SQLite/Chroma and by mocking the embedding model
where a real one isn't needed.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Run each test with CWD pointed at an isolated temp dir for data/db files."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    yield tmp_path


@pytest.fixture
def sample_text() -> str:
    return (
        "Revenue Growth\n\n"
        "The company reported revenue of $482 million for Q3, an increase of 14% "
        "year-over-year.\n\n"
        "Profitability\n\n"
        "Gross margin improved to 61.2%, up from 58.7% in the prior year quarter."
    )
