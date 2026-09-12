"""Reusable Gemini agent with key rotation and same-model retry.

Every later phase should call `complete()` or `complete_structured()` on this
class. Key rotation, 429/quota handling, 503 retry on the configured model(s),
and AllKeysExhausted live here so pipelines stay free of provider details.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
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


# Brief pause before retrying the same (or next) model on 503/5xx so a demand
# spike is not burned through attempts in under a second.
SERVER_ERROR_RETRY_SLEEP_S = 5.0


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

    if status == 429 or "429" in text or "resource exhausted" in lowered or "rate" in lowered:
        daily = "perday" in compact or "requestsperday" in compact
        if daily:
            return FailureKind.QUOTA_EXHAUSTED, None
        return FailureKind.RATE_LIMITED, retry_after
    if status in {401, 403} or "api key" in lowered or "permission" in lowered:
        return FailureKind.AUTH_INVALID, None
    if (
        status in {404, 500, 502, 503, 504}
        or "unavailable" in lowered
        or "not_found" in compact
        or "notfound" in compact
        or "no longer available" in lowered
    ):
        return FailureKind.SERVER_ERROR, retry_after
    if "timeout" in lowered or "timed out" in lowered:
        return FailureKind.TIMEOUT, None
    return None, None


def _parse_retry_after(text: str) -> int | None:
    patterns = (
        r"please retry in (\d+(?:\.\d+)?)\s*s",
        r"retryDelay['\":\s]+(\d+(?:\.\d+)?)s",
        r"retry[- ]after[:\s]*(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return int(float(match.group(1)) * 1000) + 500
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
        self._skip_min_interval = False
        # Optional Phase1DayReport (or any object with record_http).
        self.day_report = day_report

    def _configured_models(self) -> list[str]:
        models = [m for m in (self.settings.gemini_models or []) if m]
        if not models and self.settings.gemini_model:
            models = [self.settings.gemini_model]
        return models or ["gemini-3.6-flash"]

    def _models_for_call(self, model: str | None) -> list[str]:
        if model:
            return [model]
        return self._configured_models()

    def _retry_addon(self, schema: type[BaseModel] | None) -> str:
        name = getattr(schema, "__name__", "") if schema is not None else ""
        if name in {"MentionsFill", "MentionsFillExhaustive"}:
            return MENTIONS_FILL_RETRY
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
            self._wait_min_interval()
            try:
                logger.info("Gemini model %s on %s", current_model, key.id)
                try:
                    try:
                        text = self._call_once(key, prompt, sys, current_model, schema)
                    except StructuredOutputError:
                        # Empty / truncated replies still used a successful HTTP slot.
                        self._record_http(key, 200)
                        raise
                    self._record_http(key, 200)
                finally:
                    self._mark_call()
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
                if kind == FailureKind.SERVER_ERROR:
                    self._record_http(key, 503)
                elif kind in {FailureKind.RATE_LIMITED, FailureKind.QUOTA_EXHAUSTED}:
                    self._record_http(key, 429)
                logger.warning("Gemini call failed on %s (%s): %s", key.id, current_model, exc)
                if kind == FailureKind.SERVER_ERROR:
                    # With a single configured model (3.6-flash), retry the same
                    # model after a pause. Multi-model lists still rotate if set.
                    unavailable.add(current_model)
                    still = [m for m in models if m not in unavailable]
                    msg = str(exc).lower()
                    is_demand = (
                        "503" in msg
                        or "unavailable" in msg
                        or "high demand" in msg
                        or "502" in msg
                        or "504" in msg
                    )
                    if not still:
                        unavailable.clear()
                        still = [current_model]
                    wait_s = 0.0
                    if is_demand:
                        wait_s = (
                            (retry_ms / 1000.0)
                            if retry_ms
                            else SERVER_ERROR_RETRY_SLEEP_S
                        )
                        time.sleep(wait_s)
                    logger.warning(
                        "Retrying Gemini model %s%s",
                        still[0],
                        f" after {wait_s:.1f}s" if wait_s else "",
                    )
                    self._skip_min_interval = True
                    continue
                self.pool.report_failure(key, kind, retry_ms)
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

    def _wait_min_interval(self) -> None:
        if self._skip_min_interval:
            self._skip_min_interval = False
            return
        interval_ms = int(getattr(self.settings, "gemini_min_interval_ms", 0) or 0)
        if interval_ms <= 0 or self._last_call_at is None:
            return
        wait_s = interval_ms / 1000.0 - (time.monotonic() - self._last_call_at)
        if wait_s <= 0:
            return
        logger.info("Waiting %.1fs before next Gemini call", wait_s)
        time.sleep(wait_s)

    def _mark_call(self) -> None:
        self._last_call_at = time.monotonic()

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
                model=model or self.settings.gemini_model,
                schema=schema,
            )
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


def _finish_reason_name(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    finish = getattr(candidates[0], "finish_reason", None)
    if finish is None:
        return ""
    return str(getattr(finish, "name", finish))
