"""Shared NVIDIA NIM client factory, backend selection and JSON helper.

Every pattern in this repo talks to NVIDIA NIM through its OpenAI-compatible
endpoint. This module is the single place where credentials, base URL, model
and *backend* are resolved, so switching models, pointing the whole lab at a
self-hosted NIM container, or running every pattern offline is a one-line
change.

Backends (``NIM_BACKEND`` env var, or the ``main.py`` flags):

- ``live``   (default) the real NIM endpoint. Needs ``NVIDIA_API_KEY``.
- ``mock``   replays a cassette (``src/replay.py``): no key, no network,
             deterministic. ``main.py <pattern> --offline`` selects it.
- ``record`` calls the live endpoint and saves every exchange to a cassette
             (``NIM_RECORD`` / ``main.py --record PATH``).

Tracing (``NIM_TRACE=path.jsonl`` or ``main.py --trace``) wraps whichever
backend is active and logs every chat call with latency and token usage
(``src/trace.py``).

``get_client()`` returns ONE process-wide client, so nested patterns (the
orchestrator's workers) share the same backend, cassette and trace.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
CASSETTE_DIR = REPO_ROOT / "cassettes"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
BACKENDS = ("live", "mock", "record")
MEMORY_TRACE = ":memory:"


class BackendError(RuntimeError):
    """The backend itself is unusable (for example, a cassette no longer
    matches the pattern's prompts). Patterns must let this propagate instead
    of converting it into a tool or worker error message."""


_MISSING_KEY_MESSAGE = """
NVIDIA_API_KEY is not set.

Getting a free key takes about two minutes:

  1. Go to https://build.nvidia.com and sign up (free, includes credits).
  2. Open any model card and click "Get API Key".
  3. Copy the key -- it starts with "nvapi-".
  4. In this repo: copy .env.example to .env and paste the key:

       NVIDIA_API_KEY=nvapi-...

Then re-run the command.

No key yet? Every pattern also runs offline from a recorded cassette:

  python main.py react --offline
"""

_PLACEHOLDER_KEY_MESSAGE = """
NVIDIA_API_KEY still holds the placeholder value from .env.example.

Open .env and replace nvapi-XXXXXXXX... with your real key from
https://build.nvidia.com ("Get API Key" on any model card). No request was
sent: the placeholder would only have produced a 401 from the server.

No key yet? Every pattern also runs offline from a recorded cassette:

  python main.py react --offline
"""

_settings: dict[str, Any] = {}
_client: Any = None
_console_checked = False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def configure(
    *,
    backend: str | None = None,
    cassette: str | Path | None = None,
    record_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    client: Any = None,
    meta: dict | None = None,
) -> None:
    """Select the backend programmatically (main.py and the tests use this).

    Anything left as None falls back to the matching environment variable
    (NIM_BACKEND, NIM_CASSETTE, NIM_RECORD, NIM_TRACE). ``client`` injects a
    ready-made client (a fake, or the inner client for record mode). Calling
    configure() drops the current process-wide client.
    """
    global _settings, _client
    _settings = {
        key: value
        for key, value in {
            "backend": backend,
            "cassette": cassette,
            "record_path": record_path,
            "trace_path": trace_path,
            "client": client,
            "meta": meta,
        }.items()
        if value is not None
    }
    _client = None


def reset() -> None:
    """Forget any configuration and the cached client."""
    configure()


def _setting(name: str, env_var: str) -> Any:
    value = _settings.get(name)
    if value is None:
        value = os.environ.get(env_var, "").strip() or None
    return value


def backend() -> str:
    """Name of the active backend: live, mock or record."""
    name = str(_setting("backend", "NIM_BACKEND") or "live").strip().lower()
    if name not in BACKENDS:
        sys.stderr.write(
            f"\nUnknown NIM_BACKEND {name!r}. Use one of: {', '.join(BACKENDS)}.\n"
        )
        raise SystemExit(1)
    return name


def is_offline() -> bool:
    """True when replies come from a cassette rather than a model."""
    return backend() == "mock"


def _load_env() -> None:
    """Load the repo's own .env (and only that one) without overriding
    variables that are already set in the environment."""
    if ENV_FILE.is_file():
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)


def base_url() -> str:
    """NIM endpoint. Override with NIM_BASE_URL for a self-hosted container."""
    return os.environ.get("NIM_BASE_URL", "").strip() or NIM_BASE_URL


def get_model() -> str:
    """Return the chat model id. Override with the NIM_MODEL env var."""
    if backend() != "mock":
        _load_env()
    return os.environ.get("NIM_MODEL", "").strip() or DEFAULT_MODEL


def is_placeholder_key(api_key: str) -> bool:
    """Detect the dummy key shipped in .env.example (and similar stand-ins)."""
    key = api_key.strip()
    return bool(
        re.fullmatch(r"nvapi-[xX]*", key)
        or re.fullmatch(r"nvapi-\.+", key)
        or "<" in key
        or "XXXXXXXX" in key.upper()
    )


def resolve_cassette(value: str | Path | None) -> Path:
    """Accept a cassette path or a bare pattern name such as 'react'."""
    if value is None:
        sys.stderr.write(
            "\nThe mock backend needs a cassette: set NIM_CASSETTE to a pattern "
            "name (e.g. react) or a file path, or run: python main.py <pattern> --offline\n"
        )
        raise SystemExit(1)
    path = Path(value)
    if path.is_file():
        return path
    named = CASSETTE_DIR / f"{value}.json"
    if named.is_file():
        return named
    available = ", ".join(sorted(p.stem for p in CASSETTE_DIR.glob("*.json")))
    sys.stderr.write(f"\nCassette {str(value)!r} not found. Available: {available}\n")
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Client construction
# --------------------------------------------------------------------------


def safe_console() -> None:
    """Never crash on printing model output.

    Model replies routinely contain arrows, emoji or accented text. A Windows
    console or pipe using a legacy code page (cp1252) cannot encode those and
    print() raises UnicodeEncodeError mid-run. Replace unencodable characters
    instead of dying.
    """
    global _console_checked
    if _console_checked:
        return
    _console_checked = True
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError, AttributeError):
                pass


def _live_client() -> Any:
    _load_env()
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        sys.stderr.write(_MISSING_KEY_MESSAGE)
        raise SystemExit(1)
    if is_placeholder_key(api_key):
        sys.stderr.write(_PLACEHOLDER_KEY_MESSAGE)
        raise SystemExit(1)
    from openai import OpenAI

    return OpenAI(base_url=base_url(), api_key=api_key)


def _build_client() -> Any:
    safe_console()
    name = backend()
    if name == "mock":
        from src.replay import ReplayClient

        client: Any = ReplayClient.from_file(
            resolve_cassette(_setting("cassette", "NIM_CASSETTE"))
        )
    else:
        client = _settings.get("client") or _live_client()
        if name == "record":
            path = _setting("record_path", "NIM_RECORD")
            if not path:
                sys.stderr.write(
                    "\nThe record backend needs a destination: set NIM_RECORD=<file.json> "
                    "or run: python main.py <pattern> --record <file.json>\n"
                )
                raise SystemExit(1)
            from src.replay import RecordingClient

            meta = {"model": get_model(), **(_settings.get("meta") or {})}
            client = RecordingClient(client, path, meta=meta)

    trace = _setting("trace_path", "NIM_TRACE")
    if trace:
        from src.trace import TracingClient

        client = TracingClient(client, None if str(trace) == MEMORY_TRACE else trace)
    return client


def get_client() -> Any:
    """Return the process-wide chat client for the active backend.

    Live mode exits with a friendly message (not a stack trace) if the key is
    missing or still the .env.example placeholder, because that is the very
    first thing every new user hits.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def active_client() -> Any:
    """The client built so far in this process, or None."""
    return _client


def find_layer(cls: type) -> Any:
    """Find a wrapper of type ``cls`` in the active client chain (the
    tracer, the recorder or the replayer), or None."""
    layer = _client
    while layer is not None:
        if isinstance(layer, cls):
            return layer
        layer = getattr(layer, "inner", None)
    return None


# --------------------------------------------------------------------------
# Helpers every pattern uses
# --------------------------------------------------------------------------


def chat(
    client: Any,
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
    if not getattr(response, "choices", None):
        return ""
    return (response.choices[0].message.content or "").strip()


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*")


def extract_json(text: Any) -> dict | None:
    """Return the first JSON object found in a model reply, or None.

    Tolerates ``` fences, prose before the object, and commentary after it
    (even commentary that contains braces, which defeats a greedy regex).
    """
    if not isinstance(text, str):
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
