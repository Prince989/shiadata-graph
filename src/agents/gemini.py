"""Reusable Gemini 3.5 Flash agent.

Every later phase should call `complete()` or `complete_structured()` on this
class. Key rotation, 429/quota handling, and AllKeysExhausted live here so
pipelines stay free of provider details.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
    "\n\nYour previous JSON was truncated or invalid. Emit COMPLETE valid JSON only. "
    "Copy each Arabic hadith once. Keep Persian and English faithful and concise. "
    "Do not repeat sentences or pad fields."
)


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
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    lowered = text.lower()
    retry_after = _parse_retry_after(text)

    if status == 429 or "429" in text or "resource exhausted" in lowered or "rate" in lowered:
        compact = lowered.replace("_", "").replace("-", "")
        daily = "perday" in compact or "requestsperday" in compact
        if daily:
            return FailureKind.QUOTA_EXHAUSTED, None
        return FailureKind.RATE_LIMITED, retry_after
    if status in {401, 403} or "api key" in lowered or "permission" in lowered:
        return FailureKind.AUTH_INVALID, None
    if status in {500, 502, 503, 504} or "unavailable" in lowered:
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
    ):
        self.settings = settings or get_settings()
        self.pool = key_pool or KeyPool(state, self.settings)
        self._generate_fn = generate_fn

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
        attempts = max(self.settings.gemini_max_attempts, self.pool.pool_size())
        for attempt in range(attempts):
            try:
                key = self.pool.acquire()
            except AllKeysExhausted:
                raise
            sys = system
            if attempt and schema is not None:
                sys = (system or "") + COMPACT_JSON_RETRY
            try:
                text = self._call_once(key, prompt, sys, model, schema)
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
                self.pool.report_failure(key, kind, retry_ms)
                logger.warning("Gemini call failed on %s: %s", key.id, exc)
                if kind == FailureKind.QUOTA_EXHAUSTED and self.pool.all_daily_quota_locked():
                    raise AllKeysExhausted(FREE_TIER_TODAY) from exc
                continue
        if isinstance(last_error, (StructuredOutputError, json.JSONDecodeError, ValidationError)):
            raise StructuredOutputError(str(last_error)) from last_error
        kind, _ = classify_provider_error(last_error) if last_error else (None, None)
        if kind == FailureKind.QUOTA_EXHAUSTED:
            raise AllKeysExhausted(FREE_TIER_TODAY) from last_error
        raise ProviderServerError(str(last_error) if last_error else "gemini failed")

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
