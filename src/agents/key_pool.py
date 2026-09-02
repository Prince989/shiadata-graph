"""Round-robin Gemini key pool with SQLite-backed cooldowns.

This is the only place that decides which GOOGLE_API_KEY* is live. Phase 1,
dedup verification, and edge classification all go through GeminiAgent, which
calls acquire() here. Future phases should do the same — do not instantiate
google.genai.Client with a raw env key.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum

from config.settings import Settings, get_settings
from src.agents.errors import AllKeysExhausted
from src.state_manager import StateManager

logger = logging.getLogger(__name__)


class FailureKind(str, Enum):
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_INVALID = "auth_invalid"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"


COOLING_KINDS = {
    FailureKind.RATE_LIMITED,
    FailureKind.QUOTA_EXHAUSTED,
    FailureKind.AUTH_INVALID,
    FailureKind.TIMEOUT,
    FailureKind.SERVER_ERROR,
}


@dataclass(frozen=True)
class LlmKey:
    id: str
    secret: str
    index: int


def key_id_for(secret: str, index: int) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:10]
    return f"gemini-{index}-{digest}"


class KeyPool:
    def __init__(
        self,
        state: StateManager,
        settings: Settings | None = None,
        keys: list[str] | None = None,
    ):
        self.settings = settings or get_settings()
        self.state = state
        secrets = keys if keys is not None else self.settings.google_api_keys
        self._keys = [
            LlmKey(id=key_id_for(secret, i), secret=secret, index=i)
            for i, secret in enumerate(secrets)
        ]
        self._rr = 0
        if not self._keys:
            logger.warning("No Google API keys configured")
        elif len(self._keys) == 1:
            logger.warning(
                "Only one distinct Gemini key. Rotation is failover only, not extra quota."
            )
        else:
            logger.info("Loaded %d distinct Gemini key(s)", len(self._keys))

    def pool_size(self) -> int:
        return len(self._keys)

    def acquire(self) -> LlmKey:
        now = int(time.time() * 1000)
        healthy = [k for k in self._keys if self._is_healthy(k.id, now)]
        if not healthy:
            raise AllKeysExhausted(
                "All Gemini keys are cooling or disabled. State is saved; resume later."
            )
        chosen = healthy[self._rr % len(healthy)]
        self._rr += 1
        return chosen

    def report_success(self, key: LlmKey) -> None:
        self.state.clear_cooldown(key.id)

    def report_failure(
        self,
        key: LlmKey,
        kind: FailureKind,
        retry_after_ms: int | None = None,
    ) -> None:
        if kind not in COOLING_KINDS:
            return
        previous = self.state.get_cooldown(key.id)
        strikes = int(previous["strikes"]) + 1 if previous else 1
        cooldown = self._cooldown_ms(kind, strikes, retry_after_ms)
        retry_at = int(time.time() * 1000) + cooldown
        self.state.set_cooldown(key.id, kind.value, strikes, retry_at)
        logger.warning(
            "Key %s cooling for %s (%d ms, strike %d)",
            key.id,
            kind.value,
            cooldown,
            strikes,
        )

    def _is_healthy(self, key_id: str, now_ms: int) -> bool:
        record = self.state.get_cooldown(key_id)
        if not record:
            return True
        return int(record["retry_at_ms"]) <= now_ms

    def _cooldown_ms(
        self,
        kind: FailureKind,
        strikes: int,
        retry_after_ms: int | None,
    ) -> int:
        cfg = self.settings
        if kind == FailureKind.RATE_LIMITED and retry_after_ms:
            return min(retry_after_ms, cfg.key_cooldown_max_ms)
        if kind == FailureKind.QUOTA_EXHAUSTED:
            return cfg.key_quota_cooldown_ms
        if kind == FailureKind.AUTH_INVALID:
            return cfg.key_quota_cooldown_ms * 4
        exponential = cfg.key_cooldown_base_ms * (2 ** (strikes - 1))
        ceiling = (
            min(cfg.key_cooldown_max_ms, 5 * 60_000)
            if kind in {FailureKind.TIMEOUT, FailureKind.SERVER_ERROR}
            else cfg.key_cooldown_max_ms
        )
        return min(exponential, ceiling)
