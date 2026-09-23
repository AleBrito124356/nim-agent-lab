"""Small helpers shared by the test modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CASSETTES = ROOT / "cassettes"


class NetworkBlocked(RuntimeError):
    """Raised by the autouse no_network fixture on any connection attempt."""


def tool_call(call_id: str, name: str, arguments) -> dict:
    """One tool call for a scripted reply."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": call_id, "name": name, "arguments": arguments}


def fake_tool_call(name: str, arguments, call_id: str = "call_1") -> SimpleNamespace:
    """An object shaped like openai's tool_call, for _execute_tool_call."""
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
