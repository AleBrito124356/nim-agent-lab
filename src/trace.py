"""Per-run LLM trace: one JSON line per chat call, plus a cost/latency summary.

``TracingClient`` wraps any client (live, record or mock) at the transport
level -- around ``chat.completions.create`` -- so all 12 patterns get
observability without a single line of pattern code changing. It also sees
the calls that patterns make directly (tool calling, structured output).

Each JSONL record holds: call number, the pattern function that made the call
(``debate_agents._statement``, ``MemoryAgent.turn``...), the first line of
the system prompt, the full messages, sampling params, the reply, latency,
and token usage. When a backend reports no usage (hand-authored cassettes,
some NIM models) tokens are estimated at ~4 characters per token from the
real prompt the pattern built, and the record says ``"estimated": true``.

Summarize an existing trace file:
    python -m src.trace traces/debate.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_INFRA_DIR = os.path.normcase(str(Path(__file__).resolve().parent))
_INFRA_FILES = {
    os.path.normcase(str(Path(_INFRA_DIR) / name)) for name in ("nim.py", "trace.py", "replay.py")
}

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4


# --------------------------------------------------------------------------
# Token estimation (shared with src/replay.py)
# --------------------------------------------------------------------------


def estimate_tokens(text: Any) -> int:
    """Rough token count: ~4 characters per token, 0 for empty text."""
    if not text:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_prompt_tokens(request: dict) -> int:
    """Estimate the prompt size of a chat request from its messages and tools."""
    total = 0
    for message in request.get("messages") or []:
        total += MESSAGE_OVERHEAD_TOKENS
        total += estimate_tokens(message.get("content"))
        if message.get("tool_calls"):
            total += estimate_tokens(message["tool_calls"])
    if request.get("tools"):
        total += estimate_tokens(request["tools"])
    return total


def estimate_completion_tokens(content: str | None, tool_calls: list | None) -> int:
    total = estimate_tokens(content)
    for call in tool_calls or []:
        function = getattr(call, "function", None)
        if function is not None:
            total += estimate_tokens(function.name) + estimate_tokens(function.arguments)
        elif isinstance(call, dict):
            total += estimate_tokens(call)
    return total


# --------------------------------------------------------------------------
# Tracing client
# --------------------------------------------------------------------------


def _caller() -> str:
    """Name the pattern function that triggered this call, e.g.
    'reflection_agent._critique', by walking out of the infrastructure files."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
        if filename not in _INFRA_FILES:
            module = frame.f_globals.get("__name__", "")
            if module == "__main__" or not module:
                module = Path(filename).stem
            short_module = module.rsplit(".", 1)[-1]
            code = frame.f_code
            qualname = getattr(code, "co_qualname", code.co_name)
            if "." in qualname and not qualname.startswith("<"):
                return qualname  # e.g. MemoryAgent.turn
            return f"{short_module}.{qualname}"
        frame = frame.f_back
    return "unknown"


def _system_line(messages: list[dict]) -> str:
    for message in messages or []:
        if message.get("role") == "system":
            text = str(message.get("content") or "").strip()
            return text.splitlines()[0][:100] if text else ""
    return ""


def _serialize_tool_calls(tool_calls: Any) -> list[dict] | None:
    if not tool_calls:
        return None
    return [
        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
        for tc in tool_calls
    ]


def _usage_of(response: Any, request: dict) -> dict:
    """Provider-reported usage when available, otherwise an estimate."""
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion = getattr(usage, "completion_tokens", None) if usage is not None else None
    estimated = bool(getattr(usage, "estimated", False)) if usage is not None else True
    if prompt is None or completion is None:
        message = response.choices[0].message if getattr(response, "choices", None) else None
        prompt = estimate_prompt_tokens(request)
        completion = estimate_completion_tokens(
            getattr(message, "content", None), getattr(message, "tool_calls", None)
        )
        estimated = True
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(prompt) + int(completion),
        "estimated": estimated,
    }


class _Completions:
    def __init__(self, owner: "TracingClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)


class _Chat:
    def __init__(self, owner: "TracingClient") -> None:
        self.completions = _Completions(owner)


class TracingClient:
    """Wrap a chat client; record every call in memory and, optionally, JSONL."""

    def __init__(self, inner: Any, path: str | Path | None = None) -> None:
        self.inner = inner
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self.chat = _Chat(self)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")  # one trace per run

    def _create(self, **kwargs: Any) -> Any:
        record: dict[str, Any] = {
            "call": len(self.records) + 1,
            "caller": _caller(),
            "system": _system_line(kwargs.get("messages") or []),
            "model": kwargs.get("model"),
            "params": {
                key: kwargs[key]
                for key in ("temperature", "max_tokens", "stop", "tool_choice", "response_format")
                if key in kwargs
            },
            "tools": [t.get("function", {}).get("name") for t in kwargs.get("tools") or []] or None,
            "messages": list(kwargs.get("messages") or []),  # snapshot, the list keeps growing
        }
        started = time.perf_counter()
        try:
            response = self.inner.chat.completions.create(**kwargs)
        except Exception as exc:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["usage"] = {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": False,
            }
            self._write(record)
            raise
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        message = response.choices[0].message if getattr(response, "choices", None) else None
        record["response"] = {
            "content": getattr(message, "content", None),
            "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
        }
        record["usage"] = _usage_of(response, kwargs)
        self._write(record)
        return response

    def _write(self, record: dict) -> None:
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def summary(self) -> dict:
        return summarize(self.records)

    def format_summary(self, title: str = "LLM trace") -> str:
        text = format_summary(self.summary(), title=title)
        if self.path is not None:
            text += f"\n  trace file: {self.path}"
        return text


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def summarize(records: list[dict]) -> dict:
    """Aggregate call count, tokens and latency, overall and per caller."""
    by_caller: dict[str, dict] = {}
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0,
              "errors": 0, "estimated": False}
    for record in records:
        usage = record.get("usage") or {}
        row = by_caller.setdefault(
            record.get("caller", "unknown"),
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0},
        )
        for bucket in (row, totals):
            bucket["calls"] += 1
            bucket["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            bucket["completion_tokens"] += int(usage.get("completion_tokens", 0))
            bucket["latency_ms"] += float(record.get("latency_ms", 0.0))
        if record.get("error"):
            totals["errors"] += 1
        if usage.get("estimated"):
            totals["estimated"] = True
    return {"totals": totals, "by_caller": by_caller}


def format_summary(summary: dict, title: str = "LLM trace") -> str:
    totals, by_caller = summary["totals"], summary["by_caller"]
    calls = totals["calls"]
    mean = totals["latency_ms"] / calls if calls else 0.0
    lines = [
        f"{title}: {calls} call(s), {totals['prompt_tokens']:,} prompt + "
        f"{totals['completion_tokens']:,} completion tokens, "
        f"{totals['latency_ms']:,.1f} ms total latency ({mean:,.1f} ms/call)",
    ]
    if by_caller:
        width = max(28, *(len(name) for name in by_caller))
        lines.append(f"  {'caller':<{width}}  calls   prompt  completion    avg ms")
        for name, row in by_caller.items():
            avg = row["latency_ms"] / row["calls"] if row["calls"] else 0.0
            lines.append(
                f"  {name:<{width}}  {row['calls']:>5}  {row['prompt_tokens']:>7,}  "
                f"{row['completion_tokens']:>10,}  {avg:>8,.1f}"
            )
    if totals["errors"]:
        lines.append(f"  {totals['errors']} call(s) raised an error")
    if totals["estimated"]:
        lines.append("  (token counts estimated at ~4 chars/token where the backend reported none)")
    return "\n".join(lines)


def load_records(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> int:
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print("usage: python -m src.trace TRACE.jsonl [TRACE.jsonl ...]")
        return 2
    for path in paths:
        print(format_summary(summarize(load_records(path)), title=str(path)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
