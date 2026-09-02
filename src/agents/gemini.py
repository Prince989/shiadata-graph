"""Reusable Gemini 3.5 Flash agent.

Every later phase should call `complete()` or `complete_structured()` on this
class. Key rotation, 429/quota handling, and AllKeysExhausted live here so
pipelines stay free of provider details.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from config.settings import Settings, get_settings
from src.agents.errors import (
    AllKeysExhausted,
    AuthInvalid,
    ProviderServerError,
    QuotaExhausted,
    RateLimited,
    StructuredOutputError,
)

from src.agents.key_pool import FailureKind, KeyPool, LlmKey
from src.state_manager import StateManager

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


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
        if "quota" in lowered:
            return FailureKind.QUOTA_EXHAUSTED, retry_after
        return FailureKind.RATE_LIMITED, retry_after
    if status in {401, 403} or "api key" in lowered or "permission" in lowered:
        return FailureKind.AUTH_INVALID, None
    if status in {500, 502, 503, 504} or "unavailable" in lowered:
        return FailureKind.SERVER_ERROR, retry_after
    if "timeout" in lowered or "timed out" in lowered:
        return FailureKind.TIMEOUT, None
    return None, None


def _parse_retry_after(text: str) -> int | None:
    match = re.search(r"retry[- ]after[:\s]*(\d+)", text, re.I)
    if match:
        return int(match.group(1)) * 1000
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
        for _ in range(self.settings.gemini_max_attempts):
            try:
                key = self.pool.acquire()
            except AllKeysExhausted:
                raise
            try:
                text = self._call_once(key, prompt, system, model, schema)
                self.pool.report_success(key)
                return text
            except StructuredOutputError:
                raise
            except Exception as exc:  # noqa: BLE001 — classified below
                last_error = exc
                kind, retry_ms = classify_provider_error(exc)
                if kind is None:
                    raise
                self.pool.report_failure(key, kind, retry_ms)
                logger.warning("Gemini call failed on %s: %s", key.id, exc)
                continue
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
        client = genai.Client(api_key=key.secret)
        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
        response = client.models.generate_content(
            model=model or self.settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None)
        if not text:
            raise StructuredOutputError("empty Gemini response")
        return text
