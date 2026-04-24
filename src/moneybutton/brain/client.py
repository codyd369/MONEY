"""Multi-provider LLM wrapper via LiteLLM (SPEC §3 LLM, §10.5, §12, §14).

The LLM is a supporting actor here, not a primary signal source:
  - news_relevance scoring (high volume, cheap model)
  - content writing, topic discovery, weekly self-review (general model)
  - market-mapping suggestions for strategies 2 & 3

LiteLLM lets you switch providers by just changing the model string. The
prefix picks the backend:

    anthropic/claude-opus-4-7       -> Anthropic (reads ANTHROPIC_API_KEY)
    anthropic/claude-haiku-4-5
    gemini/gemini-2.5-flash         -> Google AI Studio (GEMINI_API_KEY)
    gemini/gemini-2.0-flash-exp     (free tier)
    groq/llama-3.3-70b-versatile    -> Groq (GROQ_API_KEY)
    openrouter/<model>              -> OpenRouter (OPENROUTER_API_KEY)
    ollama/llama3.1                 -> Local Ollama (OLLAMA_BASE_URL)

Settings.llm_model_general + llm_model_news use this format directly.
One process can call multiple providers in the same run — for example,
route news_relevance to free Gemini Flash and content writing to
Anthropic Opus.

What this wrapper adds on top of LiteLLM:
  - On-disk prompt cache keyed by (model, system, messages) so dev
    iteration does not burn credits.
  - Exponential-backoff retry on rate-limit / transient errors.
  - Uniform usage accounting (prompt/completion/cache tokens + elapsed_s)
    across providers.
  - Optional json_mode that appends a strict-JSON nudge and parses the
    response (still caller's job to validate shape).

Secrets live in Settings (SecretStr). This module never logs them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from moneybutton.core.config import Settings, get_settings

log = logging.getLogger("moneybutton.brain")

# LiteLLM emits a lot of debug noise by default; quiet it.
litellm.drop_params = True  # providers that don't support a param will silently ignore instead of error
litellm.suppress_debug_info = True

_RETRIABLE = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    InternalServerError,
    ServiceUnavailableError,
    APIError,  # broad catch-all for 5xx-ish
)


@dataclass
class LLMUsage:
    """Token + cost accounting. Recorded to audit after every call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class LLMResponse:
    text: str
    json_: dict | list | None
    usage: LLMUsage
    model: str
    raw_stop_reason: str | None = None


@dataclass
class BrainClient:
    """Provider-agnostic wrapper around LiteLLM's completion API."""

    settings: Settings = field(default_factory=get_settings)
    cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.settings.data_dir) / "brain_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._configure_provider_env()

    # ------------------------------------------------------------------
    def _configure_provider_env(self) -> None:
        """Surface configured API keys into os.environ so LiteLLM's built-in
        env discovery picks them up. LiteLLM reads ANTHROPIC_API_KEY,
        GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY directly.
        """
        mapping = {
            "ANTHROPIC_API_KEY": self.settings.anthropic_api_key,
            "GEMINI_API_KEY": self.settings.gemini_api_key,
            "GROQ_API_KEY": self.settings.groq_api_key,
            "OPENROUTER_API_KEY": self.settings.openrouter_api_key,
        }
        for name, secret in mapping.items():
            value = secret.get_secret_value() if secret else ""
            if value and not os.environ.get(name):
                os.environ[name] = value
        if self.settings.ollama_base_url and not os.environ.get("OLLAMA_API_BASE"):
            os.environ["OLLAMA_API_BASE"] = self.settings.ollama_base_url

    @staticmethod
    def _cache_key(model: str, system: str, messages: list[dict]) -> str:
        blob = json.dumps(
            {"model": model, "system": system, "messages": messages}, sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        use_cache: bool = True,
        json_mode: bool = False,
        mock_response: str | None = None,
    ) -> LLMResponse:
        """One-shot completion.

        `model` defaults to settings.llm_model_general. `mock_response` (dev/
        test only) bypasses any network call and returns the string verbatim.
        """
        chosen_model = model or self.settings.llm_model_general

        if json_mode:
            messages = list(messages)
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {
                    **messages[-1],
                    "content": messages[-1]["content"] + "\n\nReturn STRICT JSON only.",
                }

        # Build the message list LiteLLM expects (OpenAI-style: system goes
        # as role=system, litellm translates for Anthropic/Gemini/Ollama).
        messages_for_llm = [{"role": "system", "content": system}, *messages]

        cache_path: Path | None = None
        if use_cache:
            key = self._cache_key(chosen_model, system, messages)
            cache_path = self.cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text("utf-8"))
                cached_usage = dict(cached.get("usage", {}))
                cached_usage.pop("model", None)  # avoid dup kwarg; we fix it below
                usage = LLMUsage(model=chosen_model, **cached_usage)
                return LLMResponse(
                    text=cached["text"],
                    json_=cached.get("json"),
                    usage=usage,
                    model=chosen_model,
                    raw_stop_reason=cached.get("stop_reason"),
                )

        started = time.monotonic()
        resp = self._call_with_retry(
            model=chosen_model,
            messages=messages_for_llm,
            max_tokens=max_tokens,
            temperature=temperature,
            mock_response=mock_response,
        )
        elapsed = time.monotonic() - started

        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        text = ""
        stop_reason = None
        if choice is not None:
            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            if msg is not None:
                text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or ""
            stop_reason = getattr(choice, "finish_reason", None)
            if stop_reason is None and isinstance(choice, dict):
                stop_reason = choice.get("finish_reason")

        parsed = None
        if json_mode:
            parsed = _try_parse_json(text)
            if parsed is None:
                log.warning("LLM json_mode requested but response wasn't valid JSON")

        usage_obj = getattr(resp, "usage", None)
        usage = LLMUsage(
            model=chosen_model,
            input_tokens=_getattr_or_key(usage_obj, "prompt_tokens", 0),
            output_tokens=_getattr_or_key(usage_obj, "completion_tokens", 0),
            cache_creation_input_tokens=_getattr_or_key(
                usage_obj, "cache_creation_input_tokens", 0
            ),
            cache_read_input_tokens=_getattr_or_key(
                usage_obj, "cache_read_input_tokens", 0
            ),
            elapsed_s=elapsed,
        )

        if cache_path is not None:
            cache_path.write_text(
                json.dumps(
                    {
                        "text": text,
                        "json": parsed,
                        "usage": usage.as_dict(),
                        "stop_reason": stop_reason,
                    },
                    default=str,
                ),
                encoding="utf-8",
            )

        return LLMResponse(
            text=text,
            json_=parsed,
            usage=usage,
            model=chosen_model,
            raw_stop_reason=stop_reason,
        )

    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        wait=wait_exponential(multiplier=2, min=2, max=32),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_with_retry(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        mock_response: str | None,
    ):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if mock_response is not None:
            kwargs["mock_response"] = mock_response
        started = time.monotonic()
        try:
            return litellm.completion(**kwargs)
        finally:
            log.debug("brain.complete elapsed=%.3fs model=%s", time.monotonic() - started, model)


def _try_parse_json(text: str) -> dict | list | None:
    """Best-effort JSON parser tolerant of common LLM output quirks:
    - markdown code fences (```json ... ``` or ``` ... ```)
    - leading/trailing prose around the JSON object/array
    - Unicode smart quotes that occasionally sneak in
    """
    if not text:
        return None
    s = text.strip()

    # Strip markdown fences. LLMs (especially Gemini) wrap JSON in ```json...```.
    if s.startswith("```"):
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()

    # Try as-is first.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Find an embedded object or array — common when models add prose.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        first = s.find(open_c)
        last = s.rfind(close_c)
        if first != -1 and last > first:
            candidate = s[first : last + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _getattr_or_key(obj: Any, name: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(name)
    else:
        v = getattr(obj, name, None)
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ------------------------------- helpers ---------------------------------
def load_prompt(name: str) -> str:
    """Read a prompt markdown file from brain/prompts/ by name (without .md)."""
    here = Path(__file__).resolve().parent / "prompts" / f"{name}.md"
    return here.read_text("utf-8")


def render_prompt(name: str, **vars: Any) -> str:
    """str.format-style interpolation. Double curlies ({{,}}) must be
    pre-escaped in the template if literal braces are needed."""
    return load_prompt(name).format(**vars)
