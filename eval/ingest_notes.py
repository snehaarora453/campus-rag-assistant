"""
Bulk-ingest every file in data/notes/ directly into ChromaDB, bypassing the
API. Useful for setting up the eval test set without clicking through the UI.

Usage (from eval/ directory, with backend's venv active):
    python ingest_notes.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import rag_chain  # noqa: E402
from config import NOTES_DIR  # noqa: E402


def main():
    notes_dir = Path(NOTES_DIR)
    files = [f for f in notes_dir.glob("*") if f.suffix.lower() in (".pdf", ".txt", ".md")]

    if not files:
        print(f"No .pdf/.txt/.md files found in {notes_dir}")
        return

    for f in files:
        try:
            n = rag_chain.ingest_file(str(f), source_name=f.name)
            print(f"Indexed {f.name}: {n} chunks")
        except Exception as e:
            print(f"Failed on {f.name}: {e}")


if __name__ == "__main__":
    main()
