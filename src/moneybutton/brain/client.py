"""Anthropic SDK wrapper (SPEC §3 LLM section, §10.5, §12, §14).

The LLM is a supporting actor in this system, not a primary signal source:
  - news_relevance scoring (high volume, Haiku 4.5 by default)
  - content writing, topic discovery, weekly self-review (Opus 4.7 by default)
  - market-mapping suggestions for strategies 2 & 3

This wrapper exists so every caller gets uniform:
  - retry on 429/500/overloaded (tenacity exponential backoff),
  - structured JSON output via response_format + schema validation,
  - on-disk prompt cache keyed by (model, system, messages_hash) so dev
    iteration does not burn credits,
  - cost accounting: usage totals captured per call for the weekly report.

Secrets live in Settings (SecretStr); the client never logs them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from moneybutton.core.config import Settings, get_settings

log = logging.getLogger("moneybutton.brain")

# Exceptions considered retriable. Rate limits and transient overload must be
# retried with backoff; invalid-request and auth errors should fail fast.
_RETRIABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,  # broader catch-all for 5xx
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
    """Thin, opinionated wrapper around the Anthropic Messages API.

    Disk cache: JSON files keyed by a sha256 of the request; lookup is O(1)
    via filename. Cache is per-project under data/brain_cache/ and safe to
    wipe.
    """

    settings: Settings = field(default_factory=get_settings)
    cache_dir: Path = field(init=False)
    _client: anthropic.Anthropic | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.settings.data_dir) / "brain_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is not None:
            return self._client
        key = self.settings.anthropic_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env before running LLM calls."
            )
        self._client = anthropic.Anthropic(api_key=key)
        return self._client

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
    ) -> LLMResponse:
        """One-shot completion. `json_mode=True` appends a 'return JSON' nudge
        to the last user turn and json-loads the response; caller still must
        validate the shape.
        """
        chosen_model = model or self.settings.llm_model_general

        if json_mode:
            messages = list(messages)
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {
                    **messages[-1],
                    "content": messages[-1]["content"] + "\n\nReturn STRICT JSON only.",
                }

        if use_cache:
            key = self._cache_key(chosen_model, system, messages)
            cache_path = self.cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text("utf-8"))
                usage = LLMUsage(model=chosen_model, **cached.get("usage", {}))
                parsed = cached.get("json")
                return LLMResponse(
                    text=cached["text"],
                    json_=parsed,
                    usage=usage,
                    model=chosen_model,
                    raw_stop_reason=cached.get("stop_reason"),
                )

        resp = self._call_with_retry(
            model=chosen_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        parsed = None
        if json_mode:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                log.warning("LLM json_mode requested but response wasn't valid JSON")

        usage = LLMUsage(
            model=chosen_model,
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(
                resp.usage, "cache_creation_input_tokens", 0
            ) or 0,
            cache_read_input_tokens=getattr(
                resp.usage, "cache_read_input_tokens", 0
            ) or 0,
        )

        if use_cache:
            key = self._cache_key(chosen_model, system, messages)
            cache_path = self.cache_dir / f"{key}.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "text": text,
                        "json": parsed,
                        "usage": usage.as_dict(),
                        "stop_reason": getattr(resp, "stop_reason", None),
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
            raw_stop_reason=getattr(resp, "stop_reason", None),
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
        system: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> anthropic.types.Message:
        client = self._ensure_client()
        started = time.monotonic()
        try:
            return client.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            log.debug("brain.complete elapsed=%.3fs model=%s", time.monotonic() - started, model)


# ------------------------------- helpers ---------------------------------
def load_prompt(name: str) -> str:
    """Read a prompt markdown file from brain/prompts/ by name (without .md)."""
    here = Path(__file__).resolve().parent / "prompts" / f"{name}.md"
    return here.read_text("utf-8")


def render_prompt(name: str, **vars: Any) -> str:
    """str.format-style interpolation. Double curlies ({{,}}) must be
    pre-escaped in the template if literal braces are needed."""
    return load_prompt(name).format(**vars)
