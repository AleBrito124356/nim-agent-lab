from __future__ import annotations

import io
import json
import sys

import pytest

from src.patterns import human_in_the_loop as hitl
from src.patterns import orchestrator as orch
from src.replay import CassetteMismatch
from tests.helpers import read_jsonl

# --------------------------------------------------------------- human-in-the-loop


def test_parse_actions_keeps_valid_and_explains_discards():
    reply = json.dumps({"actions": [
        {"tool": "calculator", "args": {"expression": "1+1"}, "reason": "math"},
        "write_note",                                                   # audit: crashed with AttributeError
        {"tool": "post_to_slack", "args": {}},
        {"name": "send_email", "arguments": '{"to": "a@b.c", "subject": "s", "body": "b"}'},
        {"tool": "write_note", "args": ["not", "a", "dict"]},
    ]})
    actions, discarded = hitl._parse_actions(reply)
    assert [a["tool"] for a in actions] == ["calculator", "send_email"]
    assert actions[1]["args"]["to"] == "a@b.c"
    whys = [d["why"] for d in discarded]
    assert whys[0].startswith("not an action object")
    assert whys[1] == "unknown tool 'post_to_slack'"
    assert whys[2].startswith("args for write_note must be an object")


def test_parse_actions_limit_and_garbage():
    many = json.dumps({"actions": [{"tool": "calculator", "args": {"expression": "1"}}] * 8})
    actions, discarded = hitl._parse_actions(many)
    assert len(actions) == hitl.MAX_ACTIONS and len(discarded) == 2
    assert hitl._parse_actions('{"actions": "none"}') == ([], [])
    assert hitl._parse_actions("no json") == ([], [])


def test_note_names_are_sanitized_into_the_workspace():
    result = hitl.tool_write_note("../../etc/passwd", "x")
    assert (hitl.WORKSPACE / "etcpasswd.txt").read_text(encoding="utf-8") == "x"
    assert "Wrote 1 chars" in result
    assert hitl.tool_write_note("!!!", "x").startswith("Error")


def test_execute_reports_missing_args_and_tool_errors(monkeypatch):
    assert hitl._execute("send_email", {"to": "x"}).startswith("Error: missing argument(s)")
    monkeypatch.setitem(hitl.TOOLS, "calculator", (lambda expression: 1 / 0, ["expression"]))
    assert hitl._execute("calculator", {"expression": "1"}) == "Error: division by zero"


PLAN = json.dumps({"actions": [
    {"tool": "write_note", "args": {"name": "plan", "content": "hello"}, "reason": "r1"},
    {"tool": "calculator", "args": {"expression": "40 * 52 * 0.2"}, "reason": "r2"},
    {"tool": "send_email", "args": {"to": "t@example.com", "subject": "s", "body": "b"}, "reason": "r3"},
    {"tool": "rm_rf", "args": {}, "reason": "oops"},
]})


def test_dry_run_executes_nothing_and_audits_everything(scripted):
    scripted(PLAN)
    summary = hitl.run("goal")
    entries = read_jsonl(hitl.AUDIT_LOG)
    assert [e["decision"] for e in entries] == [
        "discarded", "auto-skipped (non-interactive)", "auto-skipped (non-interactive)",
        "auto-skipped (non-interactive)",
    ]
    assert entries[0]["why"] == "unknown tool 'rm_rf'" and all("ts" in e for e in entries)
    assert not hitl.WORKSPACE.exists()
    assert "discarded proposal: unknown tool 'rm_rf'" in summary


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_interactive_approve_edit_reject(scripted, monkeypatch):
    scripted(PLAN)
    monkeypatch.setattr(sys, "stdin", FakeTTY())
    answers = iter(["a", "e", '{"expression": "2 + 2"}', "r"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    hitl.run("goal")
    entries = read_jsonl(hitl.AUDIT_LOG)[1:]
    assert [e["decision"] for e in entries] == ["approved", "approved-with-edits", "rejected"]
    assert entries[1]["final_args"] == {"expression": "2 + 2"} and entries[1]["result"] == "4"
    assert entries[2]["result"] is None
    assert (hitl.WORKSPACE / "plan.txt").read_text(encoding="utf-8") == "hello"


def test_closed_stdin_on_a_tty_skips_instead_of_crashing(scripted, monkeypatch):
    # Windows: `< /dev/null` is the NUL device, which claims isatty() == True
    scripted(PLAN)
    monkeypatch.setattr(sys, "stdin", FakeTTY())

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    hitl.run("goal")
    decisions = [e["decision"] for e in read_jsonl(hitl.AUDIT_LOG)[1:]]
    assert decisions == ["auto-skipped (reviewer input closed)"] * 3


# --------------------------------------------------------------- orchestrator


def test_parse_subtasks_tolerates_model_variants():
    reply = json.dumps({"subtasks": [
        "convert 120 miles to km",                                       # audit: crashed before
        {"worker": "tools", "task": "multiply by 22"},
        {"agent": "writer", "description": "write bullets"},
        {"worker": "astrologer", "task": "predict fuel prices"},
        {"worker": "react"},
        {"worker": "debate", "task": "tabs or spaces?"},
    ]})
    assert orch._parse_subtasks(reply) == [
        {"worker": "direct", "task": "convert 120 miles to km"},
        {"worker": "react", "task": "multiply by 22"},
        {"worker": "reflection", "task": "write bullets"},
        {"worker": "direct", "task": "predict fuel prices"},
    ]


@pytest.mark.parametrize("reply", ["no json", '{"subtasks": "x"}', '{"tasks": []}'])
def test_parse_subtasks_garbage(reply):
    assert orch._parse_subtasks(reply) == []


def test_worker_failure_is_captured_and_merge_still_runs(scripted, monkeypatch):
    def broken(task):
        raise ValueError("worker exploded")

    monkeypatch.setitem(orch.WORKERS, "direct", broken)
    client = scripted(json.dumps({"subtasks": [{"worker": "direct", "task": "t"}]}), "merged")
    assert orch.run("goal") == "merged"
    assert "Worker failed: ValueError: worker exploded" in client.requests[1]["messages"][-1]["content"]


def test_backend_errors_propagate_through_workers(scripted, monkeypatch):
    def drifted(task):
        raise CassetteMismatch("prompt changed")

    monkeypatch.setitem(orch.WORKERS, "direct", drifted)
    scripted(json.dumps({"subtasks": [{"worker": "direct", "task": "t"}]}))
    with pytest.raises(CassetteMismatch):
        orch.run("goal")


def test_debate_worker_runs_the_real_debate_pattern(scripted, capsys):
    client = scripted(
        json.dumps({"subtasks": [{"worker": "debate", "task": "Tabs or spaces?"}]}),
        "pro opening", "con opening", "pro rebuttal", "con rebuttal", "verdict: spaces",
        "Merged: spaces.",
    )
    assert orch.run("settle it") == "Merged: spaces."
    assert client.position == 7
    assert "verdict: spaces" in client.requests[6]["messages"][-1]["content"]
    assert "[worker 1/1: debate]" in capsys.readouterr().out


def test_unparseable_supervisor_reply_falls_back_to_direct(scripted):
    client = scripted("I think we should just answer.", "direct answer", "merged")
    assert orch.run("goal") == "merged"
    assert client.requests[1]["messages"][0]["content"] == "Answer the task directly and concisely."
