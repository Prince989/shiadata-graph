"""Project paths and environment loading."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_EPUBS_DIR = DATA_DIR / "raw_epubs"
OUTPUT_DIR = DATA_DIR / "output"
STATE_DB_PATH = DATA_DIR / "state.db"
BOOKS_YAML = CONFIG_DIR / "books.yaml"
ONTOLOGY_YAML = CONFIG_DIR / "base_ontology.yaml"
LOCAL_ENV = PROJECT_ROOT / ".env"
DEFAULT_RAG_ENV = PROJECT_ROOT.parent / "shiadata-rag" / ".env"
