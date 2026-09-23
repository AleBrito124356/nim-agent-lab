from __future__ import annotations

import json

import pytest

from src.patterns import tool_calling_agent as tc
from tests.helpers import fake_tool_call, tool_call


def test_exchange_rate_table():
    data = json.loads(tc.get_exchange_rate("usd", " eur "))
    assert data["rate"] == 0.92 and data["base"] == "USD" and data["quote"] == "EUR"
    assert json.loads(tc.get_exchange_rate("EUR", "USD"))["rate"] == pytest.approx(1 / 0.92, rel=1e-5)
    assert "error" in json.loads(tc.get_exchange_rate("USD", "BTC"))


def test_current_time_offset_and_validation():
    data = json.loads(tc.get_current_time(-5))
    assert data["iso"].endswith("-05:00")
    assert "error" in json.loads(tc.get_current_time(20))
    assert "error" in json.loads(tc.get_current_time("soon"))
    assert json.loads(tc.get_current_time("-5"))["iso"].endswith("-05:00")


@pytest.mark.parametrize(
    "name,arguments,fragment",
    [
        ("get_exchange_rate", "{not json", "malformed arguments"),
        ("get_exchange_rate", "[1, 2]", "must be a JSON object"),
        ("get_exchange_rate", '"USD"', "must be a JSON object"),
        ("get_exchange_rate", '{"base": "USD"}', "bad arguments"),
        ("get_exchange_rate", '{"base": "USD", "quote": "EUR", "extra": 1}', "bad arguments"),
        ("calculator", '{"expression": "1e300**2"}', "too large"),
        ("delete_database", "{}", "unknown tool"),
    ],
)
def test_execute_tool_call_always_returns_json_error(name, arguments, fragment):
    out = tc._execute_tool_call(fake_tool_call(name, arguments))
    assert fragment in json.loads(out)["error"]


def test_wrong_argument_type_no_longer_crashes():
    # audit: {"base": 5} raised AttributeError ('int' has no attribute 'upper')
    out = json.loads(tc._execute_tool_call(fake_tool_call("get_exchange_rate", '{"base": 5, "quote": "EUR"}')))
    assert "error" in out


def test_tool_exception_becomes_json_error(monkeypatch):
    def broken(expression):
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(tc.TOOL_REGISTRY, "calculator", broken)
    out = json.loads(tc._execute_tool_call(fake_tool_call("calculator", '{"expression": "1"}')))
    assert out["error"] == "calculator failed: RuntimeError: disk on fire"


def test_dict_arguments_are_accepted():
    out = json.loads(tc._execute_tool_call(fake_tool_call("calculator", {"expression": "2*21"})))
    assert out == {"result": 42}


def test_parallel_calls_all_execute_and_are_answered_in_order(scripted):
    client = scripted(
        {"tool_calls": [
            tool_call("a", "get_exchange_rate", {"base": "USD", "quote": "EUR"}),
            tool_call("b", "get_exchange_rate", {"base": "USD", "quote": "JPY"}),
            tool_call("c", "calculator", {"expression": "1/0"}),
        ]},
        "done",
    )
    assert tc.run("rates please") == "done"
    second = client.requests[1]["messages"]
    tool_messages = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b", "c"]
    assert json.loads(tool_messages[2]["content"]) == {"error": "division by zero"}
    assistant = [m for m in second if m["role"] == "assistant"][0]
    assert len(assistant["tool_calls"]) == 3
    assert client.requests[0]["tools"] == tc.TOOL_SCHEMAS


def test_round_budget(scripted, capsys):
    reply = {"tool_calls": [tool_call("x", "calculator", {"expression": "1+1"})]}
    client = scripted(*[reply] * tc.MAX_ROUNDS)
    assert tc.run("never ends") == "The agent hit its round limit before producing a final answer."
    assert client.position == tc.MAX_ROUNDS
    assert "round budget exhausted" in capsys.readouterr().out
