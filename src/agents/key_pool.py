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
from src.agents.errors import AllKeysExhausted, FREE_TIER_TODAY
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
        n = len(self._keys)
        if n == 0:
            raise AllKeysExhausted("No Gemini API keys configured.")
        max_wait = int(getattr(self.settings, "key_acquire_wait_max_ms", 120_000))
        deadline = int(time.time() * 1000) + max_wait
        while True:
            now = int(time.time() * 1000)
            if self.all_daily_quota_locked():
                raise AllKeysExhausted(FREE_TIER_TODAY)
            for _ in range(n):
                candidate = self._keys[self._rr % n]
                self._rr += 1
                if self._is_healthy(candidate.id, now):
                    logger.info(
                        "Using Gemini key %d/%d (%s)",
                        candidate.index + 1,
                        n,
                        candidate.id,
                    )
                    return candidate
            retry_at = self._soonest_waitable_retry_at()
            if retry_at is None:
                raise AllKeysExhausted(FREE_TIER_TODAY)
            wait_ms = retry_at - now
            if wait_ms <= 0:
                continue
            if now + wait_ms > deadline:
                break
            logger.info("All Gemini keys cooling; waiting %s ms", wait_ms)
            time.sleep(wait_ms / 1000.0)
        raise AllKeysExhausted(
            "All Gemini keys are cooling or disabled. State is saved; resume later."
        )

    def all_daily_quota_locked(self) -> bool:
        now = int(time.time() * 1000)
        if not self._keys:
            return False
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record:
                return False
            if record["reason"] != FailureKind.QUOTA_EXHAUSTED.value:
                return False
            if int(record["retry_at_ms"]) <= now:
                return False
        return True

    def _soonest_waitable_retry_at(self) -> int | None:
        """Soonest retry among keys that are not on a 24h daily-quota cooldown."""
        times: list[int] = []
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record:
                return None
            if record["reason"] == FailureKind.QUOTA_EXHAUSTED.value:
                continue
            times.append(int(record["retry_at_ms"]))
        return min(times) if times else None

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
            return min(max(int(retry_after_ms), 1_000), cfg.key_cooldown_max_ms)
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
