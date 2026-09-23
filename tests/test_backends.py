"""src/nim.py backend selection, src/replay.py replay + record."""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

import pytest

from src import nim
from src.patterns import react_agent, structured_output_agent
from src.replay import (
    CassetteExhausted,
    CassetteMismatch,
    RecordingClient,
    ReplayClient,
    ReplayError,
    build_response,
)
from tests.helpers import CASSETTES, NetworkBlocked, tool_call

# --------------------------------------------------------------- nim


def test_network_is_blocked_during_tests():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("integrate.api.nvidia.com", 443))


def test_missing_key_exits_with_setup_help(capsys):
    nim.configure(backend="live")
    with pytest.raises(SystemExit) as exc:
        nim.get_client()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "NVIDIA_API_KEY is not set" in err and "--offline" in err


@pytest.mark.parametrize("key", ["nvapi-XXXXXXXXXXXXXXXXXXXXXXXX", "nvapi-", "nvapi-...", "<your key>"])
def test_placeholder_key_is_rejected_before_any_request(monkeypatch, capsys, key):
    # audit: the .env.example placeholder built a client and produced a remote 401
    monkeypatch.setenv("NVIDIA_API_KEY", key)
    nim.configure(backend="live")
    with pytest.raises(SystemExit) as exc:
        nim.get_client()
    assert exc.value.code == 1
    assert "placeholder" in capsys.readouterr().err


def test_real_looking_key_builds_a_client_without_network(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-" + "a1B2" * 12)
    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8000/v1")
    nim.configure(backend="live")
    client = nim.get_client()
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"
    assert nim.get_client() is client  # one process-wide client


def test_dotenv_is_read_from_the_repo_root_only(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    (parent / "repo").mkdir(parents=True)
    (parent / ".env").write_text("NVIDIA_API_KEY=nvapi-from-a-parent-directory\n", encoding="utf-8")
    monkeypatch.setattr(nim, "ENV_FILE", parent / "repo" / ".env")
    nim.configure(backend="live")
    with pytest.raises(SystemExit):
        nim.get_client()  # the parent directory's .env must not be picked up

    (parent / "repo" / ".env").write_text("NIM_MODEL=meta/test-model\n", encoding="utf-8")
    monkeypatch.delenv("NIM_BACKEND")
    nim.reset()
    assert nim.get_model() == "meta/test-model"


def test_mock_backend_never_reads_dotenv(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("NIM_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(nim, "ENV_FILE", env)
    assert nim.backend() == "mock"
    assert nim.get_model() == nim.DEFAULT_MODEL


def test_unknown_backend_is_a_friendly_error(monkeypatch, capsys):
    monkeypatch.setenv("NIM_BACKEND", "cloud")
    with pytest.raises(SystemExit):
        nim.backend()
    assert "Unknown NIM_BACKEND 'cloud'" in capsys.readouterr().err


def test_mock_backend_from_env_vars_uses_a_named_cassette(monkeypatch):
    monkeypatch.setenv("NIM_CASSETTE", "react")
    client = nim.get_client()
    assert isinstance(client, ReplayClient) and client.source.endswith("react.json")
    assert nim.is_offline()


def test_mock_backend_without_cassette_explains_itself(capsys):
    with pytest.raises(SystemExit):
        nim.get_client()
    assert "needs a cassette" in capsys.readouterr().err


def test_unknown_cassette_lists_the_available_ones(capsys):
    with pytest.raises(SystemExit):
        nim.resolve_cassette("nope")
    assert "react" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! {"a": {"b": [1, 2]}} Hope that helps.', {"a": {"b": [1, 2]}}),
        ('{"a": 1} and a note with {braces}', {"a": 1}),       # greedy regex failed here
        ('{broken {"a": 1}', {"a": 1}),
        ("[1, 2, 3]", None),
        ("no json here", None),
        (None, None),
    ],
)
def test_extract_json(text, expected):
    assert nim.extract_json(text) == expected


def test_chat_handles_empty_choices():
    empty = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[]))))
    assert nim.chat(empty, [{"role": "user", "content": "hi"}]) == ""


# --------------------------------------------------------------- replay


def test_replay_serves_in_order_and_checks_expect():
    client = ReplayClient({"interactions": [
        {"expect": "You are A", "response": {"content": ["line 1", "line 2"]}},
        {"expect": "You are B", "response": {"content": "two"}},
    ]})
    first = client.chat.completions.create(messages=[{"role": "system", "content": "You are A."}])
    assert first.choices[0].message.content == "line 1\nline 2"
    with pytest.raises(CassetteMismatch, match="call #2 expected a system prompt containing 'You are B'"):
        client.chat.completions.create(messages=[{"role": "system", "content": "You are C."}])
    assert client.position == 1  # a mismatch does not consume the reply


def test_replay_exhaustion_names_the_extra_call():
    client = ReplayClient.scripted("only one")
    client.chat.completions.create(messages=[])
    with pytest.raises(CassetteExhausted, match="call #2"):
        client.chat.completions.create(messages=[{"role": "system", "content": "extra"}])


def test_replay_tool_calls_and_templates():
    client = ReplayClient.scripted(
        {"tool_calls": [tool_call("t1", "get_current_time", {"utc_offset_hours": -5})]},
        "It is {{tool:t1.iso}} ({{tool:t1}})",
    )
    msg = client.chat.completions.create(messages=[]).choices[0].message
    assert msg.content is None and msg.tool_calls[0].function.name == "get_current_time"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"utc_offset_hours": -5}
    tool_result = json.dumps({"iso": "2026-01-01T00:00:00-05:00"})
    text = client.chat.completions.create(
        messages=[{"role": "tool", "tool_call_id": "t1", "content": tool_result}]
    ).choices[0].message.content
    assert text == f"It is 2026-01-01T00:00:00-05:00 ({tool_result})"


def test_template_without_matching_tool_result_is_a_mismatch():
    client = ReplayClient.scripted("It is {{tool:missing.iso}}")
    with pytest.raises(CassetteMismatch, match="no such tool result"):
        client.chat.completions.create(messages=[])


def test_replay_usage_is_estimated_unless_recorded():
    client = ReplayClient({"interactions": [
        {"response": {"content": "abcd" * 10}},
        {"response": {"content": "x"}, "usage": {"prompt_tokens": 100, "completion_tokens": 7}},
    ]})
    estimated = client.chat.completions.create(messages=[{"role": "user", "content": "a" * 40}]).usage
    assert (estimated.completion_tokens, estimated.prompt_tokens, estimated.estimated) == (10, 14, True)
    recorded = client.chat.completions.create(messages=[]).usage
    assert (recorded.prompt_tokens, recorded.completion_tokens, recorded.estimated) == (100, 7, False)


@pytest.mark.parametrize(
    "content,match",
    [("{not json", "not valid JSON"), ("[]", "must be a JSON object"),
     ('{"interactions": {}}', "no 'interactions' list"), ('{"interactions": [1]}', "needs a 'response'")],
)
def test_broken_cassette_files_are_reported(tmp_path, content, match):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReplayError, match=match):
        ReplayClient.from_file(path)


# --------------------------------------------------------------- record


def test_record_then_replay_reproduces_the_run_exactly(tmp_path, capsys):
    recording = tmp_path / "react.json"
    inner = ReplayClient.from_file(CASSETTES / "react.json")  # stands in for the live endpoint
    nim.configure(backend="record", record_path=recording, client=inner,
                  meta={"pattern": "react", "goal": react_agent.DEFAULT_GOAL})
    first = react_agent.run(react_agent.DEFAULT_GOAL)
    first_out = capsys.readouterr().out

    cassette = json.loads(recording.read_text(encoding="utf-8"))
    assert cassette["source"] == "recorded" and cassette["pattern"] == "react"
    assert len(cassette["interactions"]) == 4
    assert cassette["interactions"][0]["expect"].startswith("You are a ReAct agent")
    assert cassette["interactions"][0]["request"]["stop"] == ["Observation:"]
    assert "usage" not in cassette["interactions"][0]  # estimates are never saved as real usage

    nim.configure(backend="mock", cassette=recording)
    second = react_agent.run(react_agent.DEFAULT_GOAL)
    assert second == first
    assert capsys.readouterr().out == first_out


class UsageReportingClient:
    """A fake 'live' endpoint that reports provider usage and tool calls."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if kwargs.get("tools"):
            return build_response(None, [tool_call("c1", "calculator", {"expression": "2+2"})],
                                  {"prompt_tokens": 321, "completion_tokens": 12})
        return build_response('{"ok": true}', None, {"prompt_tokens": 50, "completion_tokens": 5})


def test_recording_keeps_tool_calls_params_and_real_usage(tmp_path):
    path = tmp_path / "c.json"
    recorder = RecordingClient(UsageReportingClient(), path, meta={"pattern": "x"})
    recorder.chat.completions.create(model="m", messages=[{"role": "system", "content": "You are T.\nMore."}],
                                     tools=[{"type": "function", "function": {"name": "calculator"}}])
    recorder.chat.completions.create(model="m", messages=[], response_format={"type": "json_object"})
    entries = json.loads(path.read_text(encoding="utf-8"))["interactions"]
    assert entries[0]["expect"] == "You are T."
    assert entries[0]["request"]["tools"] == ["calculator"]
    assert entries[0]["response"]["tool_calls"][0]["name"] == "calculator"
    assert entries[0]["usage"] == {"prompt_tokens": 321, "completion_tokens": 12}
    assert entries[1]["request"]["response_format"] == {"type": "json_object"}

    replay = ReplayClient.from_file(path)
    again = replay.chat.completions.create(messages=[{"role": "system", "content": "You are T.\nMore."}])
    assert again.choices[0].message.tool_calls[0].function.arguments == '{"expression": "2+2"}'
    assert (again.usage.prompt_tokens, again.usage.estimated) == (321, False)


def test_record_mode_needs_a_destination(capsys):
    nim.configure(backend="record", client=ReplayClient.scripted("x"))
    with pytest.raises(SystemExit):
        nim.get_client()
    assert "needs a destination" in capsys.readouterr().err


# --------------------------------------------------------------- shipped cassettes


PATTERN_MODULES = {
    "react": "react_agent", "tool-calling": "tool_calling_agent", "planner-executor": "planner_executor",
    "reflection": "reflection_agent", "debate": "debate_agents", "router": "router_agent",
    "memory": "memory_agent", "guardrails": "guardrails_agent", "code-interpreter": "code_interpreter_agent",
    "structured-output": "structured_output_agent", "human-in-the-loop": "human_in_the_loop",
    "orchestrator": "orchestrator",
}


@pytest.mark.parametrize("pattern", PATTERN_MODULES)
def test_shipped_cassette_matches_its_pattern(pattern):
    import importlib

    module = importlib.import_module(f"src.patterns.{PATTERN_MODULES[pattern]}")
    cassette = json.loads((CASSETTES / f"{pattern}.json").read_text(encoding="utf-8"))
    assert cassette["format"] == 1 and cassette["pattern"] == pattern
    assert cassette["goal"] == module.DEFAULT_GOAL  # DEFAULT_GOAL edits must re-record
    assert cassette["interactions"] and all(i["expect"] for i in cassette["interactions"])


def test_structured_output_cassette_first_reply_really_fails_validation():
    cassette = json.loads((CASSETTES / "structured-output.json").read_text(encoding="utf-8"))
    first = cassette["interactions"][0]["response"]["content"]
    with pytest.raises(Exception):
        structured_output_agent.Invoice.model_validate_json(first)
