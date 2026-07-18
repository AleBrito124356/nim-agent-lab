"""Shared NVIDIA NIM client factory.

Every pattern in this repo talks to NVIDIA NIM through its OpenAI-compatible
endpoint. This module is the single place where credentials, base URL and
model selection are resolved, so switching models (or pointing the whole lab
at a self-hosted NIM container) is a one-line change.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

_MISSING_KEY_MESSAGE = """
NVIDIA_API_KEY is not set.

Getting a free key takes about two minutes:

  1. Go to https://build.nvidia.com and sign up (free, includes credits).
  2. Open any model card and click "Get API Key".
  3. Copy the key -- it starts with "nvapi-".
  4. In this repo: copy .env.example to .env and paste the key:

       NVIDIA_API_KEY=nvapi-...

Then re-run the command.
"""


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at NVIDIA NIM.

    Exits with a friendly message (not a stack trace) if the key is missing,
    because that is the very first thing every new user hits.
    """
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        sys.stderr.write(_MISSING_KEY_MESSAGE)
        raise SystemExit(1)
    return OpenAI(base_url=NIM_BASE_URL, api_key=api_key)


def get_model() -> str:
    """Return the chat model id. Override with the NIM_MODEL env var."""
    load_dotenv()
    return os.environ.get("NIM_MODEL", "").strip() or DEFAULT_MODEL


def chat(
    client: OpenAI,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> str:
    """One-shot chat completion. Returns the assistant text, stripped.

    Thin by design: patterns that need tool calls or raw response objects
    call client.chat.completions.create directly.
    """
    response = client.chat.completions.create(
        model=model or get_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()
