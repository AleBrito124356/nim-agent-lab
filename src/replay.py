"""Cassette replay and recording: run every pattern offline and deterministically.

A *cassette* is a JSON file holding the model replies one pattern run
needs, in call order::

    {
      "format": 1,
      "pattern": "react",
      "goal": "<the goal the replies were written/recorded for>",
      "source": "hand-authored" | "recorded",
      "interactions": [
        {
          "expect": "You are a ReAct agent",          # substring of the system prompt
          "response": {
            "content": ["line 1", "line 2"],           # a string, or a list of lines
            "tool_calls": [{"id": "call_1", "name": "calculator",
                            "arguments": {"expression": "2 + 2"}}]
          },
          "usage": {"prompt_tokens": 812, "completion_tokens": 64}   # optional
        }
      ]
    }

``ReplayClient`` serves those replies through the same surface the patterns
use (``client.chat.completions.create(...)`` returning objects with
``.choices[0].message.content`` / ``.tool_calls`` / ``.usage``). Before each
reply it checks that the request's system prompt contains ``expect``: if a
pattern's prompts or control flow change, the run fails with a message naming
the call, instead of silently feeding nonsense to the wrong step.

A reply may quote a live tool result from the conversation with
``{{tool:<tool_call_id>.<field>}}`` (used for the clock in tool-calling).

``RecordingClient`` wraps a real client and writes a cassette as the run
happens, so ``python main.py <pattern> --record cassettes/<pattern>.json``
turns any live session into an offline fixture.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.nim import BackendError
from src.trace import estimate_completion_tokens, estimate_prompt_tokens

CASSETTE_FORMAT = 1


class ReplayError(BackendError):
    """Base class for cassette problems."""


class CassetteExhausted(ReplayError):
    """The pattern asked for more replies than the cassette holds."""


class CassetteMismatch(ReplayError):
    """The next reply was recorded for a different prompt."""


def _system_text(messages: list[dict]) -> str:
    for message in messages or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return ""


def _join(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, list):
        return "\n".join(str(line) for line in content)
    return str(content)


def _split(content: str | None) -> Any:
    """Store multi-line text as a list of lines so cassettes stay readable."""
    if content is None or "\n" not in content:
        return content
    return content.split("\n")


_TEMPLATE_RE = re.compile(r"\{\{tool:([A-Za-z0-9_.-]+?)(?:\.([A-Za-z0-9_]+))?\}\}")


def _fill_templates(text: str | None, messages: list[dict], call_no: int) -> str | None:
    if not text or "{{tool:" not in text:
        return text

    def lookup(match: re.Match) -> str:
        call_id, field = match.group(1), match.group(2)
        for message in messages:
            if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
                content = str(message.get("content") or "")
                if field is None:
                    return content
                try:
                    return str(json.loads(content)[field])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise CassetteMismatch(
                        f"call #{call_no}: template {match.group(0)} could not read "
                        f"field {field!r} from tool result {content!r} ({exc})"
                    ) from None
        raise CassetteMismatch(
            f"call #{call_no}: template {match.group(0)} refers to tool call "
            f"{call_id!r}, but no such tool result is in the conversation"
        )

    return _TEMPLATE_RE.sub(lookup, text)


def build_response(
    content: str | None,
    tool_calls: list[dict] | None,
    usage: dict,
    *,
    model: str = "replay",
    finish_reason: str | None = None,
    response_id: str = "replay",
) -> SimpleNamespace:
    """Build an object shaped like an openai ChatCompletion."""
    calls = None
    if tool_calls:
        calls = []
        for i, call in enumerate(tool_calls, 1):
            arguments = call.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                SimpleNamespace(
                    id=str(call.get("id") or f"call_{i}"),
                    type="function",
                    function=SimpleNamespace(name=str(call["name"]), arguments=arguments),
                )
            )
    message = SimpleNamespace(role="assistant", content=content, tool_calls=calls)
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
    )
    usage_ns = SimpleNamespace(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["prompt_tokens"] + usage["completion_tokens"],
        estimated=usage.get("estimated", False),
    )
    return SimpleNamespace(id=response_id, model=model, choices=[choice], usage=usage_ns)


class _Completions:
    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)


class _Chat:
    def __init__(self, owner: Any) -> None:
        self.completions = _Completions(owner)


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


class ReplayClient:
    """Serve cassette replies in order through chat.completions.create()."""

    def __init__(self, cassette: dict, source: str = "<memory>") -> None:
        interactions = cassette.get("interactions")
        if not isinstance(interactions, list):
            raise ReplayError(f"{source}: cassette has no 'interactions' list")
        for i, entry in enumerate(interactions, 1):
            if not isinstance(entry, dict) or not isinstance(entry.get("response"), dict):
                raise ReplayError(f"{source}: interaction #{i} needs a 'response' object")
        self.cassette = cassette
        self.interactions = interactions
        self.source = source
        self.position = 0
        self.requests: list[dict] = []
        self.chat = _Chat(self)

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayClient":
        path = Path(path)
        try:
            cassette = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{path}: not valid JSON ({exc})") from None
        if not isinstance(cassette, dict):
            raise ReplayError(f"{path}: a cassette must be a JSON object")
        return cls(cassette, source=str(path))

    @classmethod
    def scripted(cls, *replies: Any) -> "ReplayClient":
        """Quick in-memory client for tests: each reply is a string, or a dict
        with 'content' and/or 'tool_calls'. No expect checks."""
        interactions = [
            {"response": reply if isinstance(reply, dict) else {"content": reply}}
            for reply in replies
        ]
        return cls({"interactions": interactions}, source="<scripted>")

    @property
    def total(self) -> int:
        return len(self.interactions)

    @property
    def remaining(self) -> int:
        return len(self.interactions) - self.position

    def _create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        call_no = self.position + 1
        system = _system_text(messages)
        if self.position >= len(self.interactions):
            raise CassetteExhausted(
                f"{self.source}: the pattern made call #{call_no} but the cassette "
                f"holds only {len(self.interactions)} replies (system prompt of the "
                f"extra call: {system[:80]!r}). Re-record it with --record."
            )
        entry = self.interactions[self.position]
        expect = str(entry.get("expect") or "")
        if expect and expect not in system:
            raise CassetteMismatch(
                f"{self.source}: call #{call_no} expected a system prompt containing "
                f"{expect!r} but got {system[:80]!r}. The pattern's prompts or control "
                "flow changed since this cassette was made; re-record it with --record."
            )
        self.position += 1
        # Snapshot: patterns keep appending to the same messages list.
        self.requests.append({**kwargs, "messages": list(messages)})

        response = entry["response"]
        content = _fill_templates(_join(response.get("content")), messages, call_no)
        tool_calls = response.get("tool_calls") or None

        result = build_response(
            content,
            tool_calls,
            {"prompt_tokens": 0, "completion_tokens": 0},
            model=str(kwargs.get("model") or self.cassette.get("model") or "replay"),
            finish_reason=response.get("finish_reason"),
            response_id=f"replay-{call_no}",
        )
        recorded = entry.get("usage")
        if isinstance(recorded, dict) and {"prompt_tokens", "completion_tokens"} <= set(recorded):
            prompt, completion, estimated = (
                int(recorded["prompt_tokens"]), int(recorded["completion_tokens"]), False
            )
        else:
            # Hand-authored replies carry no provider usage: estimate from the
            # real prompt the pattern built, and say so.
            message = result.choices[0].message
            prompt = estimate_prompt_tokens(kwargs)
            completion = estimate_completion_tokens(message.content, message.tool_calls)
            estimated = True
        result.usage = SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            estimated=estimated,
        )
        return result


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------


def interaction_from(request: dict, response: Any) -> dict:
    """Turn one live request/response pair into a cassette interaction."""
    system = _system_text(request.get("messages") or []).strip()
    message = response.choices[0].message
    entry: dict[str, Any] = {
        "expect": system.splitlines()[0][:80] if system else "",
        "request": {
            key: request[key]
            for key in ("model", "temperature", "max_tokens", "stop", "tool_choice", "response_format")
            if key in request
        },
        "response": {"content": _split(message.content)},
    }
    if request.get("tools"):
        entry["request"]["tools"] = [t.get("function", {}).get("name") for t in request["tools"]]
    if getattr(message, "tool_calls", None):
        entry["response"]["tool_calls"] = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in message.tool_calls
        ]
    finish = getattr(response.choices[0], "finish_reason", None)
    if finish:
        entry["response"]["finish_reason"] = finish
    usage = getattr(response, "usage", None)
    if usage is not None and not getattr(usage, "estimated", False):
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is not None and completion is not None:
            entry["usage"] = {"prompt_tokens": int(prompt), "completion_tokens": int(completion)}
    return entry


class RecordingClient:
    """Pass calls through to ``inner`` and save each exchange to a cassette.

    The file is rewritten after every call, so an interrupted run still
    leaves a usable (shorter) cassette behind.
    """

    def __init__(self, inner: Any, path: str | Path, meta: dict | None = None) -> None:
        self.inner = inner
        self.path = Path(path)
        self.cassette: dict[str, Any] = {
            "format": CASSETTE_FORMAT,
            "source": "recorded",
            **(meta or {}),
            "interactions": [],
        }
        self.chat = _Chat(self)
        self._flush()

    def _create(self, **kwargs: Any) -> Any:
        response = self.inner.chat.completions.create(**kwargs)
        self.cassette["interactions"].append(interaction_from(kwargs, response))
        self._flush()
        return response

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.cassette, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
