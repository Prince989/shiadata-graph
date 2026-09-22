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
from datetime import date, datetime, timedelta, timezone
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


def key_id_for(secret: str, index: int, prefix: str = "gemini") -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{index}-{digest}"


def _pacific_tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/Los_Angeles")
    except Exception:  # noqa: BLE001 — Windows may lack tzdata
        return timezone(timedelta(hours=-7))


def pacific_day_start_ms(now_ms: int | None = None) -> int:
    """Unix ms of the most recent midnight in America/Los_Angeles."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    local = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).astimezone(_pacific_tz())
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


def pacific_calendar_date(now_ms: int | None = None) -> date:
    """Gemini free-tier day boundary: calendar date in America/Los_Angeles."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    local = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).astimezone(_pacific_tz())
    return local.date()


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
        self.provider = str(getattr(self.settings, "llm_provider", "gemini") or "gemini")
        self._key_prefix = "groq" if self.provider == "groq" else "gemini"
        if keys is not None:
            secrets = keys
        elif self.provider == "groq":
            secrets = list(self.settings.groq_api_keys)
        else:
            secrets = list(self.settings.google_api_keys)
        self._keys = [
            LlmKey(id=key_id_for(secret, i, self._key_prefix), secret=secret, index=i)
            for i, secret in enumerate(secrets)
        ]
        self._rr = 0
        label = "Groq" if self.provider == "groq" else "Gemini"
        if not self._keys:
            logger.warning("No %s API keys configured", label)
        elif len(self._keys) == 1:
            logger.warning(
                "Only one distinct %s key. Rotation is failover only, not extra quota.",
                label,
            )
        else:
            logger.info("Loaded %d distinct %s key(s)", len(self._keys), label)
        self.clear_quota_locks_for_new_pacific_day()
        self._log_blocked_summary()

    def clear_quota_locks_for_new_pacific_day(self) -> int:
        """Drop free-tier overnight locks that are not for *today's* Pacific date.

        Google RPD resets at America/Los_Angeles midnight. Locks carry
        ``exhausted_day`` (YYYY-MM-DD). Legacy rows without that field are
        cleared — they were often stale report re-locks pointing at tonight.
        """
        today = pacific_calendar_date().isoformat()
        cleared = 0
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record or record["reason"] != FailureKind.QUOTA_EXHAUSTED.value:
                continue
            day = record.get("exhausted_day")
            if day == today:
                continue
            self.state.clear_cooldown(key.id)
            cleared += 1
        if cleared:
            logger.info(
                "Cleared %d free-tier overnight lock(s) (not for Pacific day %s)",
                cleared,
                today,
            )
        return cleared

    def pool_size(self) -> int:
        return len(self._keys)

    def _log_blocked_summary(self) -> None:
        now = int(time.time() * 1000)
        auth_ns: list[int] = []
        quota_ns: list[int] = []
        rate_ns: list[tuple[int, int]] = []
        for key in self._keys:
            record = self.state.get_cooldown(key.id)
            if not record or self._is_healthy(key.id, now):
                continue
            if record["reason"] == FailureKind.AUTH_INVALID.value:
                auth_ns.append(key.index + 1)
            elif record["reason"] == FailureKind.QUOTA_EXHAUSTED.value:
                quota_ns.append(key.index + 1)
            elif record["reason"] == FailureKind.RATE_LIMITED.value:
                left = max(0, int(record["retry_at_ms"]) - now)
                rate_ns.append((key.index + 1, left))
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
        if rate_ns:
            logger.info(
                "%s key(s) still on short 429 cooldown: %s",
                "Groq" if self.provider == "groq" else "Gemini",
                ", ".join(f"{n} ({left / 1000:.0f}s left)" for n, left in rate_ns),
            )

    def acquire(self) -> LlmKey:
        n = len(self._keys)
        if n == 0:
            raise AllKeysExhausted(
                "No Groq API keys configured."
                if self.provider == "groq"
                else "No Gemini API keys configured."
            )
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
                        reason = stale["reason"]
                        # Keep rate-limit strike counts across short cooldowns and
                        # successes; only drop rows that are not 429-tracking.
                        if reason == FailureKind.RATE_LIMITED.value:
                            pass
                        elif reason == FailureKind.QUOTA_EXHAUSTED.value:
                            # Unlocked after Pacific midnight — start a fresh day.
                            self.state.clear_cooldown(candidate.id)
                        else:
                            self.state.clear_cooldown(candidate.id)
                    logger.info(
                        "Using %s key %d/%d (%s)",
                        "Groq" if self.provider == "groq" else "Gemini",
                        candidate.index + 1,
                        n,
                        candidate.id,
                    )
                    return candidate
                record = self.state.get_cooldown(candidate.id) or {}
                left_ms = max(0, int(record.get("retry_at_ms") or 0) - now)
                logger.info(
                    "Skipping %s key %d/%d (%s, %ss left)",
                    "Groq" if self.provider == "groq" else "Gemini",
                    candidate.index + 1,
                    n,
                    record.get("reason") or "cooling",
                    max(1, left_ms // 1000) if left_ms else 0,
                )
            retry_at = self._soonest_waitable_retry_at()
            if retry_at is None:
                raise AllKeysExhausted(self._unusable_stop_message(now) or FREE_TIER_TODAY)
            wait_ms = retry_at - now
            if wait_ms <= 0:
                continue
            if now + wait_ms > deadline:
                break
            logger.info(
                "All %s keys cooling; waiting %s ms",
                "Groq" if self.provider == "groq" else "Gemini",
                wait_ms,
            )
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

    def lock_labels(self) -> dict[int, str]:
        """1-based key number → short lock label; empty string if currently free."""
        now = int(time.time() * 1000)
        labels = {
            FailureKind.QUOTA_EXHAUSTED.value: "quota",
            FailureKind.AUTH_INVALID.value: "auth",
            FailureKind.RATE_LIMITED.value: "rate",
            FailureKind.TIMEOUT.value: "wait",
            FailureKind.SERVER_ERROR.value: "wait",
        }
        out: dict[int, str] = {}
        for key in self._keys:
            n = key.index + 1
            if self._is_healthy(key.id, now):
                out[n] = ""
                continue
            record = self.state.get_cooldown(key.id)
            reason = (record or {}).get("reason") or ""
            out[n] = labels.get(reason, "lock")
        return out

    def lock_indexes_for_day(
        self,
        indexes: list[int],
        *,
        strikes: int = 4,
    ) -> list[LlmKey]:
        """Overnight-lock the given 0-based pool indexes (from day-report 429s)."""
        now = int(time.time() * 1000)
        retry_at = next_pacific_midnight_ms(now)
        locked: list[LlmKey] = []
        for index in indexes:
            if index < 0 or index >= len(self._keys):
                continue
            key = self._keys[index]
            previous = self.state.get_cooldown(key.id)
            if (
                previous
                and previous["reason"] == FailureKind.QUOTA_EXHAUSTED.value
                and self._quota_still_locked(previous, now)
            ):
                continue
            use_strikes = max(strikes, int(previous["strikes"]) if previous else 0, 4)
            day = pacific_calendar_date(now).isoformat()
            self.state.set_cooldown(
                key.id,
                FailureKind.QUOTA_EXHAUSTED.value,
                use_strikes,
                retry_at,
                exhausted_day=day,
            )
            logger.warning(
                "Key %s overnight-locked from day-report 429s (strike %d)",
                key.id,
                use_strikes,
            )
            locked.append(key)
        return locked

    def report_success(self, key: LlmKey) -> None:
        """Mark the key usable but keep 429 strike counts for the Pacific day."""
        previous = self.state.get_cooldown(key.id)
        if not previous:
            return
        reason = previous["reason"]
        if reason == FailureKind.QUOTA_EXHAUSTED.value:
            # Still day-locked — ignore (should not get a success while locked).
            return
        if reason == FailureKind.RATE_LIMITED.value:
            # Preserve strikes; retry_at=now means immediately reusable.
            now = int(time.time() * 1000)
            self.state.set_cooldown(
                key.id,
                FailureKind.RATE_LIMITED.value,
                int(previous["strikes"]),
                now,
                exhausted_day=None,
            )
            return
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
        if kind == FailureKind.RATE_LIMITED:
            strikes = self._next_429_strikes(previous, now)
            # Gemini free-tier: more than three 429s this Pacific day → overnight.
            # Groq 429 is a per-minute output budget; never overnight-lock it.
            if strikes > 3 and self.provider != "groq":
                kind = FailureKind.QUOTA_EXHAUSTED
                retry_after_ms = None
        elif previous and not self._is_healthy(key.id, now):
            strikes = int(previous["strikes"]) + 1
        else:
            strikes = 1
        cooldown = self._cooldown_ms(kind, strikes, retry_after_ms)
        retry_at = int(time.time() * 1000) + cooldown
        exhausted_day = (
            pacific_calendar_date(now).isoformat()
            if kind == FailureKind.QUOTA_EXHAUSTED
            else None
        )
        self.state.set_cooldown(
            key.id, kind.value, strikes, retry_at, exhausted_day=exhausted_day
        )
        logger.warning(
            "Key %s cooling for %s (%d ms, strike %d)",
            key.id,
            kind.value,
            cooldown,
            strikes,
        )

    def _next_429_strikes(self, previous: dict | None, now_ms: int) -> int:
        """Cumulative 429 strikes for the current Pacific day (survive short cooldowns)."""
        if not previous:
            return 1
        reason = previous["reason"]
        if reason not in {
            FailureKind.RATE_LIMITED.value,
            FailureKind.QUOTA_EXHAUSTED.value,
        }:
            return 1
        if reason == FailureKind.QUOTA_EXHAUSTED.value and not self._quota_still_locked(
            previous, now_ms
        ):
            return 1
        # Rate-limit row from a previous Pacific day (stale leftover).
        if reason == FailureKind.RATE_LIMITED.value:
            day_start = pacific_day_start_ms(now_ms)
            # retry_at is last_failure + short cooldown; if that window ended
            # before today's Pacific midnight, treat as a new day.
            if int(previous["retry_at_ms"]) < day_start:
                return 1
        return int(previous["strikes"]) + 1

    def _is_healthy(self, key_id: str, now_ms: int) -> bool:
        record = self.state.get_cooldown(key_id)
        if not record:
            return True
        if record["reason"] == FailureKind.QUOTA_EXHAUSTED.value:
            return not self._quota_still_locked(record, now_ms)
        return int(record["retry_at_ms"]) <= now_ms

    def _quota_still_locked(self, record: dict, now_ms: int) -> bool:
        """True while ``exhausted_day`` equals today's Pacific calendar date.

        Free-tier RPD resets at America/Los_Angeles midnight. Legacy rows
        without ``exhausted_day`` are treated as unlocked (stale re-locks).
        """
        day = record.get("exhausted_day")
        if not day:
            return False
        return day == pacific_calendar_date(now_ms).isoformat()

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
