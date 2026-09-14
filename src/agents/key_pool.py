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
from datetime import datetime, timedelta, timezone
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


def _pacific_tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/Los_Angeles")
    except Exception:  # noqa: BLE001 — Windows may lack tzdata
        return timezone(timedelta(hours=-7))


def pacific_day_start_ms(now_ms: int) -> int:
    """Unix ms of the most recent midnight in America/Los_Angeles."""
    local = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).astimezone(_pacific_tz())
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


def next_pacific_midnight_ms(now_ms: int) -> int:
    """Unix ms of the next midnight in America/Los_Angeles (Gemini free-tier RPD reset)."""
    local = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).astimezone(_pacific_tz())
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    nxt = start + timedelta(days=1)
    return int(nxt.timestamp() * 1000)


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
        self._log_blocked_summary()

    def pool_size(self) -> int:
        return len(self._keys)

    def _log_blocked_summary(self) -> None:
        now = int(time.time() * 1000)
        auth_ns: list[int] = []
        quota_ns: list[int] = []
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record or self._is_healthy(key.id, now):
                continue
            if record["reason"] == FailureKind.AUTH_INVALID.value:
                auth_ns.append(key.index + 1)
            elif record["reason"] == FailureKind.QUOTA_EXHAUSTED.value:
                quota_ns.append(key.index + 1)
        if auth_ns:
            logger.warning(
                "Skipping %d auth-invalid key(s) until cooldown ends or state is cleared: %s",
                len(auth_ns),
                auth_ns,
            )
        if quota_ns:
            logger.info(
                "%d key(s) already on daily free-tier cooldown: %s",
                len(quota_ns),
                quota_ns if len(quota_ns) <= 12 else f"{quota_ns[:12]}…",
            )

    def acquire(self) -> LlmKey:
        n = len(self._keys)
        if n == 0:
            raise AllKeysExhausted("No Gemini API keys configured.")
        max_wait = int(getattr(self.settings, "key_acquire_wait_max_ms", 300_000))
        deadline = int(time.time() * 1000) + max_wait
        while True:
            now = int(time.time() * 1000)
            stop_msg = self._unusable_stop_message(now)
            if stop_msg:
                raise AllKeysExhausted(stop_msg)
            for _ in range(n):
                candidate = self._keys[self._rr % n]
                self._rr += 1
                if self._is_healthy(candidate.id, now):
                    stale = self.state.get_cooldown(candidate.id)
                    if stale:
                        self.state.clear_cooldown(candidate.id)
                    logger.info(
                        "Using Gemini key %d/%d (%s)",
                        candidate.index + 1,
                        n,
                        candidate.id,
                    )
                    return candidate
            retry_at = self._soonest_waitable_retry_at()
            if retry_at is None:
                raise AllKeysExhausted(self._unusable_stop_message(now) or FREE_TIER_TODAY)
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

    def _unusable_stop_message(self, now_ms: int) -> str | None:
        """If every key is daily-quota or auth-dead, return a stop message."""
        if not self._keys:
            return None
        auth = 0
        quota = 0
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record:
                return None
            reason = record["reason"]
            if reason == FailureKind.AUTH_INVALID.value:
                if int(record["retry_at_ms"]) > now_ms:
                    auth += 1
                    continue
                return None
            if reason == FailureKind.QUOTA_EXHAUSTED.value and self._quota_still_locked(
                record, now_ms
            ):
                quota += 1
                continue
            return None
        if auth and not quota:
            return (
                f"All {auth} Gemini key(s) are auth-invalid (401/403). "
                "Replace those keys in .env, clear their cooldowns in data/state.db, then resume."
            )
        if auth and quota:
            return (
                f"{quota} key(s) hit free-tier daily quota; {auth} key(s) are auth-invalid. "
                "Progress is saved. Fix auth-dead keys or wait until midnight Pacific for quota."
            )
        if quota:
            return FREE_TIER_TODAY
        return None

    def all_daily_quota_locked(self) -> bool:
        now = int(time.time() * 1000)
        if not self._keys:
            return False
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record or record["reason"] != FailureKind.QUOTA_EXHAUSTED.value:
                return False
            if not self._quota_still_locked(record, now):
                return False
        return True

    def _soonest_waitable_retry_at(self) -> int | None:
        """Soonest retry among keys on short cooldowns (not daily quota / auth)."""
        times: list[int] = []
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record:
                return None
            if record["reason"] in {
                FailureKind.QUOTA_EXHAUSTED.value,
                FailureKind.AUTH_INVALID.value,
            }:
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
        now = int(time.time() * 1000)
        previous = self.state.get_cooldown(key.id)
        if previous and not self._is_healthy(key.id, now):
            strikes = int(previous["strikes"]) + 1
        else:
            strikes = 1
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
        if record["reason"] == FailureKind.QUOTA_EXHAUSTED.value:
            return not self._quota_still_locked(record, now_ms)
        return int(record["retry_at_ms"]) <= now_ms

    def _quota_still_locked(self, record: dict, now_ms: int) -> bool:
        """True until the next Pacific midnight after the original daily-quota 429.

        Google free-tier RPD resets at midnight America/Los_Angeles, not 24h
        after the error. Older rows stored lock+24h; those unlock at PT midnight too.
        """
        retry_at = int(record["retry_at_ms"])
        if retry_at <= now_ms:
            return False
        quota_ms = int(self.settings.key_quota_cooldown_ms)
        assumed_lock = retry_at - quota_ms
        if assumed_lock < pacific_day_start_ms(now_ms):
            return False
        return now_ms < next_pacific_midnight_ms(now_ms)

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
            now = int(time.time() * 1000)
            return max(60_000, next_pacific_midnight_ms(now) - now)
        if kind == FailureKind.AUTH_INVALID:
            return cfg.key_quota_cooldown_ms * 4
        exponential = cfg.key_cooldown_base_ms * (2 ** (strikes - 1))
        ceiling = (
            min(cfg.key_cooldown_max_ms, 5 * 60_000)
            if kind in {FailureKind.TIMEOUT, FailureKind.SERVER_ERROR}
            else cfg.key_cooldown_max_ms
        )
        return min(exponential, ceiling)
