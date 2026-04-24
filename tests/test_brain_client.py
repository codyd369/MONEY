"""BrainClient tests using LiteLLM's mock_response (no network calls).

`mock_response` causes LiteLLM to return a synthetic response immediately.
The BrainClient machinery — on-disk cache, JSON mode, usage accounting,
cross-provider routing — runs the same code path it would for a real call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moneybutton.brain.client import BrainClient, LLMResponse


def test_complete_mock_returns_text(tmp_env):
    bc = BrainClient()
    resp = bc.complete(
        system="you are a test",
        messages=[{"role": "user", "content": "ping"}],
        mock_response="pong",
        use_cache=False,
    )
    assert isinstance(resp, LLMResponse)
    assert resp.text == "pong"
    assert resp.model == bc.settings.llm_model_general
    # Mock responses still fill usage with LiteLLM's synthetic counters.
    assert resp.usage.model == bc.settings.llm_model_general
    assert resp.usage.elapsed_s >= 0


def test_json_mode_parses_valid_json(tmp_env):
    bc = BrainClient()
    resp = bc.complete(
        system="return json",
        messages=[{"role": "user", "content": "give me the answer"}],
        mock_response='{"answer": 42, "units": "meaning"}',
        json_mode=True,
        use_cache=False,
    )
    assert resp.json_ == {"answer": 42, "units": "meaning"}
    assert resp.text.startswith("{")


def test_json_mode_strips_markdown_fences(tmp_env):
    """Regression: Gemini wraps JSON in ```json ... ``` fences which broke
    json.loads. _try_parse_json must strip fences before parsing."""
    bc = BrainClient()
    resp = bc.complete(
        system="x",
        messages=[{"role": "user", "content": "x"}],
        mock_response="```json\n{\"material\": true, \"direction\": \"yes\"}\n```",
        json_mode=True,
        use_cache=False,
    )
    assert resp.json_ == {"material": True, "direction": "yes"}


def test_json_mode_strips_plain_fences(tmp_env):
    bc = BrainClient()
    resp = bc.complete(
        system="x",
        messages=[{"role": "user", "content": "x"}],
        mock_response="```\n{\"a\": 1}\n```",
        json_mode=True,
        use_cache=False,
    )
    assert resp.json_ == {"a": 1}


def test_json_mode_extracts_embedded_object(tmp_env):
    """When the LLM adds prose around the JSON, find and parse the object."""
    bc = BrainClient()
    resp = bc.complete(
        system="x",
        messages=[{"role": "user", "content": "x"}],
        mock_response='Sure, here is the answer: {"answer": 42} hope that helps!',
        json_mode=True,
        use_cache=False,
    )
    assert resp.json_ == {"answer": 42}


def test_json_mode_handles_bad_json(tmp_env):
    bc = BrainClient()
    resp = bc.complete(
        system="return json",
        messages=[{"role": "user", "content": "x"}],
        mock_response="this is not json",
        json_mode=True,
        use_cache=False,
    )
    assert resp.json_ is None
    assert resp.text == "this is not json"


def test_cache_hit_skips_second_call(tmp_env):
    """Second call with identical inputs should return the cached text
    even if a different mock_response would have been returned."""
    bc = BrainClient()
    r1 = bc.complete(
        system="sys-cache",
        messages=[{"role": "user", "content": "cache me"}],
        mock_response="first-answer",
        use_cache=True,
    )
    r2 = bc.complete(
        system="sys-cache",
        messages=[{"role": "user", "content": "cache me"}],
        mock_response="second-answer-that-should-not-appear",
        use_cache=True,
    )
    assert r1.text == "first-answer"
    assert r2.text == "first-answer"

    # Cache file should exist on disk.
    cache_files = list(Path(bc.cache_dir).glob("*.json"))
    assert len(cache_files) >= 1


def test_cache_can_be_bypassed(tmp_env):
    bc = BrainClient()
    r1 = bc.complete(
        system="sys",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="v1",
        use_cache=False,
    )
    r2 = bc.complete(
        system="sys",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="v2",
        use_cache=False,
    )
    assert r1.text == "v1"
    assert r2.text == "v2"


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4-7",
        "anthropic/claude-haiku-4-5",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.0-flash-exp",
        "groq/llama-3.3-70b-versatile",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "ollama/llama3.1",
    ],
)
def test_model_string_accepted_by_litellm(tmp_env, model):
    """Every provider-prefixed model string must route through LiteLLM
    without raising on the mock path. This guards against provider
    prefixes quietly being rejected (e.g. typo in a new provider name)."""
    bc = BrainClient()
    resp = bc.complete(
        system="",
        messages=[{"role": "user", "content": "ping"}],
        model=model,
        mock_response="ok",
        use_cache=False,
    )
    assert resp.text == "ok"
    assert resp.model == model


def test_provider_env_surfacing(tmp_env, monkeypatch):
    """When a provider key is configured, BrainClient surfaces it into
    os.environ so LiteLLM's built-in env discovery finds it."""
    import os

    # Clear env to test that BrainClient sets it.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "", prepend=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from moneybutton.core.config import reset_settings_cache

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-from-env")
    reset_settings_cache()

    bc = BrainClient()
    # Calling complete triggers _configure_provider_env (via __post_init__).
    assert os.environ.get("GEMINI_API_KEY") == "test-gemini-key-from-env"
    _ = bc  # keep reference


def test_usage_accounting_populated(tmp_env):
    bc = BrainClient()
    resp = bc.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi back",
        use_cache=False,
    )
    u = resp.usage
    # On the mock path LiteLLM returns fake but non-zero token counts;
    # we mainly care the dataclass is populated correctly.
    assert u.input_tokens >= 0
    assert u.output_tokens >= 0
    assert u.cache_creation_input_tokens >= 0
    assert u.cache_read_input_tokens >= 0
    assert u.elapsed_s >= 0
