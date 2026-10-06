"""Central configuration for the Campus RAG Assistant backend."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_K = 4

CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = "campus_notes"

NOTES_DIR = str(BASE_DIR / "data" / "notes")
