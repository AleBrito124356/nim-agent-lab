from __future__ import annotations

import json

import pytest

from src.patterns import reflection_agent as rf
from src.patterns import router_agent as rt

# --------------------------------------------------------------- reflection


@pytest.mark.parametrize(
    "reply,score",
    [
        ('{"score": 8}', 8),
        ('{"score": "8/10"}', 8),
        ('{"score": 8.6}', 9),
        ('{"score": 15}', 10),
        ('{"score": -3}', 1),
        ('```json\n{"score": 7, "issues": []}\n```', 7),
        ('Here is my critique: {"score": 6} -- hope it helps {really}', 6),
    ],
)
def test_critique_score_parsing_and_clamping(reply, score):
    assert rf._parse_critique(reply)["score"] == score


def test_critique_without_usable_score_is_neutral():
    critique = rf._parse_critique('{"score": "great", "issues": "too long"}')
    assert critique["score"] == rf.NEUTRAL_SCORE
    assert critique["issues"] == ["too long", "critique had no usable score"]


def test_unparseable_critique_degrades_gracefully():
    assert rf._parse_critique("Looks fine to me!") == {
        "score": 5, "strengths": [], "issues": ["critique unparseable"], "suggestions": []
    }


def test_best_draft_is_kept_when_revisions_regress(scripted, capsys):
    scripted(
        "draft zero",
        json.dumps({"score": 7, "issues": ["x"], "suggestions": ["y"]}),
        "revision one",
        json.dumps({"score": 4, "issues": ["worse"]}),
        "revision two",
        json.dumps({"score": 6}),
    )
    assert rf.run("write", rounds=2) == "draft zero"
    assert "best score: 7/10 (the first draft)" in capsys.readouterr().out


def test_stops_early_once_target_score_is_reached(scripted):
    client = scripted("great draft", json.dumps({"score": 9}))
    assert rf.run("write") == "great draft"
    assert client.position == 2


# --------------------------------------------------------------- router


@pytest.mark.parametrize(
    "reply,route",
    [
        ('{"route": "coder", "reason": "code"}', "coder"),
        ('Route decision:\n{"route": "Writer", "reason": "prose"}', "writer"),
        ("coder", "coder"),
        ("I would send this to the analyst.", "analyst"),
        ("Either coder or writer could work.", "analyst"),   # ambiguous -> fallback
        ('{"route": "engineer"}', "analyst"),
        ("", "analyst"),
    ],
)
def test_parse_route(reply, route):
    assert rt.parse_route(reply)[0] == route


def test_route_uses_specialist_prompt_and_temperature(scripted, capsys):
    client = scripted('{"route": "coder", "reason": "needs code"}', "def merge(a, b): ...")
    assert rt.run("merge two lists") == "def merge(a, b): ..."
    assert client.requests[1]["temperature"] == rt.SPECIALISTS["coder"]["temperature"]
    assert client.requests[1]["messages"][0]["content"] == rt.SPECIALISTS["coder"]["system"]
    assert "[route] -> coder (temp 0.1)" in capsys.readouterr().out
