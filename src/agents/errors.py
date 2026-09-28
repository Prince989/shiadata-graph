"""Errors that callers of the reusable Gemini / embedding agents can catch."""


FREE_TIER_TODAY = (
    "End of Gemini free tier for today. Progress is saved; "
    "try again after midnight Pacific (about 10:30 in Tehran)."
)


class AgentError(Exception):
    """Base class for LLM and embedding agent failures."""


class AllKeysExhausted(AgentError):
    """Every Gemini key is cooling or disabled. Persist state and stop."""


class QuotaExhausted(AgentError):
    """A single key hit daily/project quota."""


class RateLimited(AgentError):
    """HTTP 429 on a single key."""


class AuthInvalid(AgentError):
    """Key rejected as invalid."""


class ProviderServerError(AgentError):
    """5xx / timeout from the provider."""


class RequestTooLarge(AgentError):
    """This prompt exceeds the provider's per-request token cap (Groq 413 / ITPM)."""


class StructuredOutputError(AgentError):
    """Model returned JSON that did not match the requested schema."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        record_http_200: bool = True,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.record_http_200 = record_http_200
