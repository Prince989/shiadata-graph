"""Reusable Gemini agent with key rotation and same-model retry.

Every later phase should call `complete()` or `complete_structured()` on this
class. Key rotation, 429/quota handling, 503 skip-to-next-key, and
AllKeysExhausted live here so pipelines stay free of provider details.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from config.settings import Settings, get_settings
from src.agents.errors import (
    AllKeysExhausted,
    AuthInvalid,
    FREE_TIER_TODAY,
    ProviderServerError,
    QuotaExhausted,
    RateLimited,
    StructuredOutputError,
)

from src.agents.key_pool import FailureKind, KeyPool, LlmKey
from src.state_manager import StateManager

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

COMPACT_JSON_RETRY = (
    "\n\nYour previous JSON failed validation. Emit COMPLETE valid JSON only.\n"
    "CRITICAL: every numbered hadith MUST include a non-empty `mentions` array "
    "(at least 1 object, prefer 3-8). Each mention is "
    "{text, type, salience, evidence}. type is one of "
    "concept|person|place|group|event|work. evidence is a verbatim matn span.\n"
    "Do NOT leave mentions as []. Continuations may omit mentions; numbered "
    "markers (1-, 2-, 3 -, …) may not.\n"
    "Copy each Arabic hadith once. Keep Persian and English faithful and concise. "
    "Do not pad or repeat sentences."
)

MENTIONS_FILL_RETRY = (
    "\n\nYour previous JSON failed validation. Return ONLY "
    '{"mentions":[...]} with at least 2 objects. Each object MUST have '
    "text, type (concept|person|place|group|event|work), salience (0-1), "
    "and evidence (verbatim matn span). Do NOT return mentions:[]. "
    "Do NOT return hadith_fa/hadith_en/ravis on this call."
)

TAFSIR_RETRY = (
    "\n\nYour previous JSON failed validation. Emit TafsirExtraction JSON only.\n"
    "Do NOT echo the unit. mentions REQUIRED (prefer 4-12). Each mention is "
    "{text, type, salience, evidence}. text is an Arabic label; evidence is a "
    "verbatim span from THIS unit. quotes are Qur'anic spans only. "
    "cited_hadiths: source_work, speaker, span, text_ar, text_fa, text_en "
    "(never leave text_ar/text_en empty on a real citation). "
    "tafsir_ar, tafsir_fa, and tafsir_en are COMPLETE translations of this unit, "
    "not summaries. tafsir_fa is fluent Persian, not a paste of the source."
)

GROQ_JSON_TAIL = (
    "\nReturn JSON only matching the requested schema. Do not wrap in markdown. "
    "Do not return the Arabic matn."
)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def classify_provider_error(exc: BaseException) -> tuple[FailureKind | None, int | None]:
    """Map SDK/HTTP errors to cooldown kinds. Schema failures return (None, None)."""
    if isinstance(exc, RateLimited):
        return FailureKind.RATE_LIMITED, None
    if isinstance(exc, QuotaExhausted):
        return FailureKind.QUOTA_EXHAUSTED, None
    if isinstance(exc, AuthInvalid):
        return FailureKind.AUTH_INVALID, None
    if isinstance(exc, StructuredOutputError):
        return None, None
    text = str(exc)
    raw_status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status = int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        status = None
    lowered = text.lower()
    compact = lowered.replace("_", "").replace("-", "").replace(" ", "")
    retry_after = _parse_retry_after(text)

    # Auth before rate/429: error bodies often include "GenerateContent", and the
    # old `"rate" in text` check matched inside that word and mis-labeled 401/403.
    auth_markers = (
        "unauthenticated",
        "permission_denied",
        "account_state_invalid",
        "api key not valid",
        "api key invalid",
        "denied access",
        "consumer_invalid",
    )
    network_markers = (
        "getaddrinfo failed",
        "name or service not known",
        "nodename nor servname",
        "failed to resolve",
        "connecterror",
        "connect error",
        "connection reset",
        "connection refused",
        "connection aborted",
        "network is unreachable",
        "temporarily unavailable",
        "errno 11001",
        "errno 11002",
        "winerror 10013",  # socket access forbidden (firewall) — not API auth
        "winerror 10054",
        "winerror 10060",
        "name resolution",
        "access a socket",
        "forbidden by its access permissions",
    )
    if any(m in lowered for m in network_markers):
        return FailureKind.TIMEOUT, retry_after or 5_000
    if status in {401, 403} or any(m in lowered for m in auth_markers):
        return FailureKind.AUTH_INVALID, None
    if (
        status == 429
        or "429" in text
        or "resource exhausted" in lowered
        or "rate limit" in lowered
        or "ratelimit" in compact
    ):
        # First hits use Google's retryDelay (rate_limited). KeyPool escalates to
        # overnight quota_exhausted after more than 3 strikes on the same key.
        return FailureKind.RATE_LIMITED, retry_after or 5_000
    if "api key" in lowered:
        return FailureKind.AUTH_INVALID, None
    if (
        status in {404, 500, 502, 503, 504}
        or "unavailable" in lowered
        or "not_found" in compact
        or "notfound" in compact
        or "no longer available" in lowered
    ):
        return FailureKind.SERVER_ERROR, retry_after
    if (
        "timeout" in lowered
        or "timed out" in lowered
    ):
        # Transient local network / DNS (common with flaky VPN). Retry with pause.
        return FailureKind.TIMEOUT, retry_after or 5_000
    return None, None


def _short_provider_error(exc: BaseException) -> str:
    """One-line Groq/Gemini error for logs (status reason, not the full JSON)."""
    text = str(exc or "").strip()
    try:
        data = json.loads(text)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            code = str(err.get("code") or "").strip()
            msg = str(err.get("message") or "").strip()
            if code and msg:
                return f"{code}: {msg[:240]}"
            return (msg or code or text)[:280]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return re.sub(r"\s+", " ", text)[:280]


def _parse_retry_after(text: str) -> int | None:
    patterns = (
        r"please retry in (\d+(?:\.\d+)?)\s*ms",
        r"please retry in (\d+(?:\.\d+)?)\s*s",
        r"retryDelay['\":\s]+(\d+(?:\.\d+)?)s",
        r"retry[- ]after[:\s]*(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            raw = float(match.group(1))
            if pattern.endswith("ms"):
                return max(int(raw) + 500, 1_000)
            return int(raw * 1000) + 500
    return None


class GeminiAgent:
    """Function-style Gemini client shared across ETL phases."""

    def __init__(
        self,
        state: StateManager,
        settings: Settings | None = None,
        key_pool: KeyPool | None = None,
        generate_fn=None,
        day_report=None,
    ):
        self.settings = settings or get_settings()
        self.pool = key_pool or KeyPool(state, self.settings)
        self._generate_fn = generate_fn
        self._last_call_at: float | None = None
        self._last_call_at_by_key: dict[str, float] = {}
        self._skip_min_interval = False
        # Optional Phase1DayReport (or any object with record_http).
        self.day_report = day_report

    def _configured_models(self) -> list[str]:
        if self._is_groq():
            model = (self.settings.groq_model or "").strip()
            return [model] if model else ["qwen/qwen3.8-27b"]
        models = [m for m in (self.settings.gemini_models or []) if m]
        if not models and self.settings.gemini_model:
            models = [self.settings.gemini_model]
        return models or ["gemini-3.6-flash"]

    def _is_groq(self) -> bool:
        return str(getattr(self.settings, "llm_provider", "gemini") or "gemini") == "groq"

    def _models_for_call(self, model: str | None) -> list[str]:
        if model:
            return [model]
        return self._configured_models()

    def _retry_addon(self, schema: type[BaseModel] | None) -> str:
        name = getattr(schema, "__name__", "") if schema is not None else ""
        if name in {"MentionsFill", "MentionsFillExhaustive"}:
            return MENTIONS_FILL_RETRY
        if name == "TafsirExtraction":
            return TAFSIR_RETRY
        return COMPACT_JSON_RETRY

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> str:
        """Plain-text completion. Use this from any future phase."""
        return self._run(prompt, system=system, model=model, schema=None)

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> T:
        """JSON completion validated into `schema`. Phase 1 and Phase 2 both use this."""
        raw = self._run(prompt, system=system, model=model, schema=schema)
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, str):
                data = json.loads(data)
            return schema.model_validate(data)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise StructuredOutputError(str(exc)) from exc

    def _run(
        self,
        prompt: str,
        *,
        system: str | None,
        model: str | None,
        schema: type[BaseModel] | None,
    ) -> str:
        last_error: BaseException | None = None
        models = self._models_for_call(model)
        unavailable: set[str] = set()
        attempts = max(
            self.settings.gemini_max_attempts,
            self.pool.pool_size(),
            len(models),
        )
        key: LlmKey | None = None
        for attempt in range(attempts):
            try:
                if key is None:
                    key = self.pool.acquire()
            except AllKeysExhausted:
                raise
            remaining = [m for m in models if m not in unavailable] or models
            current_model = remaining[0]
            sys = system
            if attempt and schema is not None:
                sys = (system or "") + self._retry_addon(schema)
                if last_error is not None:
                    # Feed the exact schema failure back so the model fixes
                    # missing mentions instead of re-emitting the same hollow JSON.
                    sys = (
                        f"{sys}\n\nPrevious attempt failed validation:\n"
                        f"{last_error}\nFix every listed error in this reply."
                    )
            self._wait_min_interval(key)
            try:
                logger.info(
                    "%s model %s on %s",
                    "Groq" if self._is_groq() else "Gemini",
                    current_model,
                    key.id,
                )
                try:
                    try:
                        text = self._call_once(key, prompt, sys, current_model, schema)
                    except StructuredOutputError:
                        # Empty / truncated replies still used a successful HTTP slot.
                        self._record_http(key, 200)
                        raise
                    self._record_http(key, 200)
                    logger.info(
                        "HTTP 200 %s key %d/%d (%s) model=%s",
                        "Groq" if self._is_groq() else "Gemini",
                        key.index + 1,
                        self.pool.pool_size(),
                        key.id,
                        current_model,
                    )
                finally:
                    self._mark_call(key)
                if schema is not None:
                    data = json.loads(text) if isinstance(text, str) else text
                    if isinstance(data, str):
                        data = json.loads(data)
                    schema.model_validate(data)
                    if not isinstance(text, str):
                        text = json.dumps(data, ensure_ascii=False)
                self.pool.report_success(key)
                return text
            except (StructuredOutputError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "structured output retry %s/%s: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — classified below
                last_error = exc
                kind, retry_ms = classify_provider_error(exc)
                if kind is None:
                    raise
                if (
                    kind == FailureKind.RATE_LIMITED
                    and self._is_groq()
                    and not retry_ms
                ):
                    retry_ms = int(
                        getattr(self.settings, "groq_min_interval_ms", 75_000) or 75_000
                    )
                if kind == FailureKind.SERVER_ERROR:
                    status = 503
                    self._record_http(key, 503)
                elif kind in {FailureKind.RATE_LIMITED, FailureKind.QUOTA_EXHAUSTED}:
                    status = 429
                    self._record_http(key, 429)
                    # Day-report 429 count is the source of truth (survives short
                    # cooldowns / successes). Lock overnight after more than 3.
                    if (
                        kind == FailureKind.RATE_LIMITED
                        and not self._is_groq()
                        and self._day_429_count(key) > 3
                    ):
                        kind = FailureKind.QUOTA_EXHAUSTED
                        retry_ms = None
                elif kind == FailureKind.AUTH_INVALID:
                    status = 401
                    self._record_http(key, 401)
                elif kind == FailureKind.TIMEOUT:
                    status = 0
                else:
                    status = 0
                logger.warning(
                    "HTTP %s %s key %d/%d (%s) model=%s: %s",
                    status or kind.value,
                    "Groq" if self._is_groq() else "Gemini",
                    key.index + 1,
                    self.pool.pool_size(),
                    key.id,
                    current_model,
                    _short_provider_error(exc),
                )
                if kind == FailureKind.SERVER_ERROR:
                    msg = str(exc).lower()
                    compact = msg.replace("_", "").replace("-", "").replace(" ", "")
                    is_capacity = (
                        "503" in msg
                        or "502" in msg
                        or "504" in msg
                        or "high demand" in msg
                        or "temporarily unavailable" in msg
                    )
                    is_retired = (
                        "404" in msg
                        or "not_found" in compact
                        or "notfound" in compact
                        or "no longer available" in msg
                    )
                    if is_retired and not is_capacity:
                        unavailable.add(current_model)
                        still = [m for m in models if m not in unavailable]
                        if not still:
                            unavailable.clear()
                            still = [current_model]
                        logger.warning("Retrying Gemini model %s", still[0])
                        self._skip_min_interval = True
                        continue
                    logger.warning(
                        "503/unavailable on %s; skipping to next key",
                        key.id,
                    )
                if kind == FailureKind.TIMEOUT:
                    wait_s = (retry_ms / 1000.0) if retry_ms else 5.0
                    logger.warning(
                        "Network/timeout on %s; waiting %.1fs then retrying",
                        key.id,
                        wait_s,
                    )
                    time.sleep(wait_s)
                self.pool.report_failure(key, kind, retry_ms)
                report = self.day_report
                if report is not None and getattr(report, "live", False):
                    flush = getattr(report, "flush", None)
                    if callable(flush):
                        flush()
                key = None
                if kind == FailureKind.QUOTA_EXHAUSTED and self.pool.all_daily_quota_locked():
                    raise AllKeysExhausted(FREE_TIER_TODAY) from exc
                continue
        if isinstance(last_error, (StructuredOutputError, json.JSONDecodeError, ValidationError)):
            raise StructuredOutputError(str(last_error)) from last_error
        kind, _ = classify_provider_error(last_error) if last_error else (None, None)
        if kind == FailureKind.QUOTA_EXHAUSTED:
            raise AllKeysExhausted(FREE_TIER_TODAY) from last_error
        raise ProviderServerError(str(last_error) if last_error else "gemini failed")

    def _record_http(self, key: LlmKey, status: int) -> None:
        report = self.day_report
        if report is None:
            return
        record = getattr(report, "record_http", None)
        if callable(record):
            record(key.index, status)

    def _day_429_count(self, key: LlmKey) -> int:
        report = self.day_report
        if report is None:
            return 0
        count = getattr(report, "count_429", None)
        if callable(count):
            return int(count(key.index))
        return 0

    def _wait_min_interval(self, key: LlmKey | None = None) -> None:
        if self._skip_min_interval:
            self._skip_min_interval = False
            return
        if self._is_groq():
            interval_ms = int(getattr(self.settings, "groq_min_interval_ms", 0) or 0)
            last_at = (
                self._last_call_at_by_key.get(key.id) if key is not None else None
            )
        else:
            interval_ms = int(getattr(self.settings, "gemini_min_interval_ms", 0) or 0)
            last_at = self._last_call_at
        if interval_ms <= 0 or last_at is None:
            return
        wait_s = interval_ms / 1000.0 - (time.monotonic() - last_at)
        if wait_s <= 0:
            return
        if self._is_groq() and key is not None:
            logger.info(
                "Waiting %.1fs before next Groq call on key %d/%d (%s)",
                wait_s,
                key.index + 1,
                self.pool.pool_size(),
                key.id,
            )
        else:
            logger.info("Waiting %.1fs before next Gemini call", wait_s)
        time.sleep(wait_s)

    def _mark_call(self, key: LlmKey | None = None) -> None:
        now = time.monotonic()
        if self._is_groq() and key is not None:
            self._last_call_at_by_key[key.id] = now
            return
        self._last_call_at = now

    def _call_once(
        self,
        key: LlmKey,
        prompt: str,
        system: str | None,
        model: str | None,
        schema: type[BaseModel] | None,
    ) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(
                key=key,
                prompt=prompt,
                system=system,
                model=model
                or (
                    self.settings.groq_model
                    if self._is_groq()
                    else self.settings.gemini_model
                ),
                schema=schema,
            )
        if self._is_groq():
            return self._call_groq(key, prompt, system, model, schema)
        previous = os.environ.get("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = key.secret
        try:
            client = genai.Client(api_key=key.secret)
            config_kwargs: dict[str, Any] = {}
            if system:
                config_kwargs["system_instruction"] = system
            if schema is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = schema
            max_tokens = int(getattr(self.settings, "gemini_max_output_tokens", 0) or 0)
            if max_tokens:
                config_kwargs["max_output_tokens"] = max_tokens
            response = client.models.generate_content(
                model=model or self.settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        finally:
            if previous is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = previous
        finish_name = _finish_reason_name(response)
        if finish_name in {"MAX_TOKENS", "LENGTH"}:
            raise StructuredOutputError(f"response truncated ({finish_name})")
        text = getattr(response, "text", None)
        if not text:
            raise StructuredOutputError("empty Gemini response")
        return text

    def _call_groq(
        self,
        key: LlmKey,
        prompt: str,
        system: str | None,
        model: str | None,
        schema: type[BaseModel] | None,
    ) -> str:
        sys = system or ""
        if schema is not None:
            sys = f"{sys}{GROQ_JSON_TAIL}"
        messages: list[dict[str, str]] = []
        if sys.strip():
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": model or self.settings.groq_model,
                "messages": messages,
                "temperature": 0.6,
                "max_completion_tokens": int(
                    getattr(self.settings, "groq_max_completion_tokens", 16_384)
                    or 16_384
                ),
                "top_p": 0.95,
                "reasoning_effort": str(
                    getattr(self.settings, "groq_reasoning_effort", "medium")
                    or "medium"
                ),
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            GROQ_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key.secret}",
                "Content-Type": "application/json",
                "User-Agent": "shiadata-graph",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.settings.gemini_timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", "replace")
            text = err_body or str(exc)
            if exc.code == 429:
                raise RateLimited(text) from exc
            if exc.code in {401, 403}:
                raise AuthInvalid(text) from exc
            if exc.code in {404, 500, 502, 503, 504}:
                raise ProviderServerError(text) from exc
            raise ProviderServerError(f"HTTP {exc.code}: {text}") from exc
        except urllib.error.URLError as exc:
            raise ProviderServerError(str(exc)) from exc
        choices = payload.get("choices") or []
        if not choices:
            raise StructuredOutputError("empty Groq response")
        choice = choices[0] or {}
        finish = str(choice.get("finish_reason") or "").lower()
        if finish in {"length", "max_tokens"}:
            raise StructuredOutputError(f"response truncated ({finish})")
        message = choice.get("message") or {}
        text = message.get("content")
        if not text:
            raise StructuredOutputError("empty Groq response")
        return str(text)


def _finish_reason_name(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    finish = getattr(candidates[0], "finish_reason", None)
    if finish is None:
        return ""
    return str(getattr(finish, "name", finish))
