from __future__ import annotations

import json

import pytest

from src.patterns import planner_executor as pe


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"steps": ["a", "b"]}', ["a", "b"]),
        ('```json\n{"steps": ["a"]}\n```', ["a"]),
        ('Here is the plan: {"steps": ["a", "b"]} Let me know {if} needed.', ["a", "b"]),
        # audit: dict steps used to become "{'step': 1, 'action': 'add'}"
        ('{"steps": [{"step": 1, "action": "add the costs"}]}', ["add the costs"]),
        ('{"steps": [{"step": "Compute venue cost"}]}', ["Compute venue cost"]),
        ('{"remaining_steps": ["b", "c"]}', ["b", "c"]),
        ('{"plan": ["x"]}', ["x"]),
        ('{"steps": ["a", 3, null, ""]}', ["a"]),
        ('{"steps": []}', []),
    ],
)
def test_parse_steps(reply, expected):
    assert pe._parse_steps(reply) == expected


@pytest.mark.parametrize("reply", ['{"result": "ok"}', "no json at all", '{"steps": "a, b"}'])
def test_parse_steps_returns_none_without_a_steps_list(reply):
    assert pe._parse_steps(reply) is None


def test_parse_steps_caps_plan_length():
    reply = json.dumps({"steps": [f"s{i}" for i in range(20)]})
    assert len(pe._parse_steps(reply)) == pe.MAX_STEPS


def test_replan_without_steps_key_keeps_the_remaining_plan(scripted, capsys):
    # audit: a valid-JSON replan without "steps" silently wiped steps B and C
    scripted(
        json.dumps({"steps": ["step A", "step B", "step C"]}),
        "did A",
        '{"note": "looks good"}',          # replan 1: no steps list -> keep plan
        "did B",
        '{"remaining_steps": ["step C"]}',  # replan 2: alias key accepted
        "did C",
        "merged answer",
    )
    assert pe.run("goal") == "merged answer"
    out = capsys.readouterr().out
    assert "[execute 2] step B" in out and "[execute 3] step C" in out
    assert "keeping the remaining plan" in out
    assert "finished after 3 step(s)" in out


def test_audit_replan_scenario_executes_b_and_c(scripted, capsys):
    scripted(
        json.dumps({"steps": ["step A", "step B", "step C"]}),
        "did A",
        json.dumps({"remaining_steps": ["step B", "step C"]}),
        "did B",
        json.dumps({"steps": ["step C"]}),
        "did C",
        "final",
    )
    pe.run("goal")
    out = capsys.readouterr().out
    assert "[execute 3] step C" in out


def test_empty_replan_ends_the_run(scripted, capsys):
    client = scripted(json.dumps({"steps": ["A", "B"]}), "did A", '{"steps": []}', "final")
    assert pe.run("goal") == "final"
    assert client.position == 4
    assert "judged the goal complete" in capsys.readouterr().out


def test_calc_line_after_prose_is_detected(scripted, capsys):
    client = scripted(json.dumps({"steps": ["add"]}), "Let me compute.\nCALC: 2 + 2", "It is 4.", "4")
    pe.run("2+2")
    assert "[calc] 2 + 2 = 4" in capsys.readouterr().out
    assert client.requests[2]["messages"][-1]["content"] == "Calculator result: 4"


def test_calculator_budget_forces_a_prose_answer(scripted):
    client = scripted(
        json.dumps({"steps": ["add"]}),
        "CALC: 1 + 1", "CALC: 2 + 2", "CALC: 3 + 3",
        "Outcome: 6.",
        "final",
    )
    pe.run("sums")
    assert "Calculator budget" in client.requests[4]["messages"][-1]["content"]


def test_unparseable_plan_falls_back_to_a_single_step(scripted, capsys):
    scripted("I would first think about it.", "direct answer", "final")
    assert pe.run("goal") == "final"
    assert "[execute 1] Answer the goal directly: goal" in capsys.readouterr().out
