"""Runtime settings. Secrets come from this project's .env, then shiadata-rag/.env."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.paths import DEFAULT_RAG_ENV, LOCAL_ENV, PROJECT_ROOT, RAW_EPUBS_DIR, STATE_DB_PATH

_GOOGLE_KEY_PATTERN = re.compile(r"^GOOGLE_API_KEY(\d*)$")


def collect_env_maps() -> dict[str, str]:
    """Merge RAG .env then local .env then os.environ (later wins)."""
    merged: dict[str, str] = {}
    rag_path = Path(os.environ.get("SHIADATA_RAG_ENV", str(DEFAULT_RAG_ENV)))
    for path in (rag_path, LOCAL_ENV):
        if path.exists():
            for key, value in dotenv_values(path, encoding="utf-8-sig").items():
                if value:
                    merged[key] = value
    for key, value in os.environ.items():
        if value:
            merged[key] = value
    return merged


def collect_google_keys(env: dict[str, str] | None = None) -> list[str]:
    sources = env or collect_env_maps()
    found: dict[int, str] = {}
    for name, value in sources.items():
        match = _GOOGLE_KEY_PATTERN.match(name)
        if not match or not value.strip():
            continue
        index = int(match.group(1)) if match.group(1) else 0
        found[index] = value.strip()
    ordered: list[str] = []
    seen: set[str] = set()
    for index in sorted(found):
        key = found[index]
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(LOCAL_ENV) if LOCAL_ENV.exists() else None,
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: str = ""
    google_api_keys: list[str] = Field(default_factory=list)
    gemini_model: str = "gemini-3.5-flash"
    embedding_model: str = "text-embedding-3-small"
    log_level: str = "INFO"

    raw_data_dir: Path = RAW_EPUBS_DIR
    state_db: Path = STATE_DB_PATH
    project_root: Path = PROJECT_ROOT

    gemini_timeout_s: float = 120.0
    gemini_max_attempts: int = 6
    key_cooldown_base_ms: int = 30_000
    key_cooldown_max_ms: int = 1_800_000
    key_quota_cooldown_ms: int = 86_400_000  # 24h — resume tomorrow
    embed_batch_size: int = 64
    history_pages_per_call: int = 50
    history_max_chars: int = 80_000
    dedup_similarity: float = 0.92
    edge_similarity: float = 0.5
    edge_group_cap: int = 200
    skip_min_chars: int = 40

    @model_validator(mode="after")
    def _load_keys(self) -> "Settings":
        env = collect_env_maps()
        if not self.openai_api_key:
            object.__setattr__(self, "openai_api_key", env.get("OPENAI_API_KEY", ""))
        if not self.google_api_keys:
            object.__setattr__(self, "google_api_keys", collect_google_keys(env))
        if env.get("GEMINI_MODEL"):
            object.__setattr__(self, "gemini_model", env["GEMINI_MODEL"])
        if env.get("EMBEDDING_MODEL"):
            object.__setattr__(self, "embedding_model", env["EMBEDDING_MODEL"])
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
