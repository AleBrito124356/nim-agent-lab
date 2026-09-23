from __future__ import annotations

import pytest

from src.patterns import react_agent as ra

# --------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('Thought: x\nAction: convert_units(42, "km", "mi")', ("convert_units", '42, "km", "mi"')),
        # audit regression: the old greedy DOTALL regex swallowed the next line
        ("Thought: first convert.\nAction: convert_units(42, km, mi)\n"
         "Thought: then I will call calculator(x ** 2)", ("convert_units", "42, km, mi")),
        ("Action: calculator((2 + 3) * 4) because precedence", ("calculator", "(2 + 3) * 4")),
        ("Thought: t\nAction: get_weather\nAction Input: Panama City", ("get_weather", "Panama City")),
        ("Action: calculator((2+3)*4", ("calculator", "(2+3)*4")),
        ("Action: get_weather(St. John's)", ("get_weather", "St. John's")),
        ("  Action:  calculator ( 1 + 1 )", ("calculator", " 1 + 1 ")),
    ],
)
def test_parse_action(reply, expected):
    assert ra.parse_action(reply) == expected


def test_parse_action_none_without_action_line():
    assert ra.parse_action("Thought: I am just thinking out loud.") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('42, "km", \'mi\'', ["42", "km", "mi"]),
        ("value=42, from_unit=km, to_unit=mi", ["42", "km", "mi"]),
        ('"a, b", c', ["a, b", "c"]),
        ("f(1, 2), 3", ["f(1, 2)", "3"]),
    ],
)
def test_split_args(raw, expected):
    assert ra.split_args(raw) == expected


def test_quoted_convert_units_call_now_works():
    # audit: convert_units(42, "km", "mi") always failed and burned the budget
    assert ra._dispatch("convert_units", '42, "km", "mi"') == "26.0976 mi"
    assert ra._dispatch("convert_units", "42, 'km', 'mi'") == "26.0976 mi"


def test_dispatch_strips_quotes_and_keywords_for_single_argument_tools():
    assert ra._dispatch("calculator", '"26.0976 ** 2"') == "681.08472576"
    assert ra._dispatch("calculator", "expression=2 ** 3") == "8"
    assert ra._dispatch("get_weather", 'city="Panama City"').startswith("Panama City:")


def test_dispatch_rejects_unknown_tool_and_wrong_arity():
    assert ra._dispatch("search_web", "x").startswith("Error: unknown tool")
    assert ra._dispatch("convert_units", "42, km").startswith("Error: convert_units expects")


def test_truncate_hallucinated_observation():
    reply = "Thought: x\nAction: calculator(2+2)\nObservation: 5\nFinal Answer: 5"
    assert ra._truncate_observation(reply) == "Thought: x\nAction: calculator(2+2)"


# --------------------------------------------------------------- tools


@pytest.mark.parametrize(
    "args,expected",
    [
        (("42", "km", "mi"), "26.0976 mi"),
        (("100", "c", "f"), "212 f"),
        (("32", "F", "C"), "0 c"),
        (("1", "gal", "l"), "3.78541 l"),
        (("10", "lb", "kg"), "4.53592 kg"),
    ],
)
def test_convert_units(args, expected):
    assert ra.convert_units(*args) == expected


@pytest.mark.parametrize(
    "args,fragment",
    [
        (("1", "km", "kg"), "cannot convert"),
        (("1", "km", "parsec"), "unsupported unit"),
        (("abc", "km", "mi"), "is not a number"),
        (("inf", "km", "mi"), "not a finite number"),
    ],
)
def test_convert_units_errors(args, fragment):
    assert fragment in ra.convert_units(*args)


def test_weather_is_deterministic_and_case_insensitive():
    first = ra.get_weather("Panama City")
    assert first == ra.get_weather("Panama City")
    assert first == "Panama City: 18 C, sunny, humidity 85% (demo data)"
    assert ra.get_weather("panama city").split(":", 1)[1] == first.split(":", 1)[1]
    assert ra.get_weather("   ").startswith("Error")


# --------------------------------------------------------------- loop


def test_run_happy_path(scripted):
    client = scripted(
        'Thought: convert\nAction: convert_units(42, "km", "mi")',
        "Thought: done\nFinal Answer: about 26.1 miles",
    )
    assert ra.run("convert 42 km") == "about 26.1 miles"
    assert client.position == 2
    assert client.requests[1]["messages"][-1]["content"] == "Observation: 26.0976 mi"
    assert client.requests[0]["stop"] == ["Observation:"]


def test_run_stops_at_iteration_budget(scripted, capsys):
    client = scripted(*["Thought: again\nAction: calculator(1 + 1)"] * ra.MAX_ITERATIONS)
    result = ra.run("loop forever")
    assert result == "The agent hit its iteration limit before producing a final answer."
    assert client.position == ra.MAX_ITERATIONS
    assert "iteration budget exhausted" in capsys.readouterr().out


def test_format_error_is_reported_back_and_recovered(scripted):
    client = scripted("I think it is 4.", "Thought: fix format\nFinal Answer: 4")
    assert ra.run("2+2") == "4"
    assert "no valid 'Action: tool(args)' line" in client.requests[1]["messages"][-1]["content"]


def test_self_written_observation_is_discarded_and_tool_runs(scripted):
    client = scripted(
        "Thought: x\nAction: calculator(2 + 2)\nObservation: 5\nFinal Answer: 5",
        "Final Answer: 4",
    )
    assert ra.run("2+2") == "4"
    assert client.requests[1]["messages"][-1]["content"] == "Observation: 4"


def test_final_answer_after_an_action_waits_for_the_real_observation(scripted):
    client = scripted("Action: calculator(6 * 7)\nFinal Answer: 41", "Final Answer: 42")
    assert ra.run("6*7") == "42"
    assert client.requests[1]["messages"][-1]["content"] == "Observation: 42"


def test_overflowing_tool_call_does_not_crash_the_loop(scripted):
    # audit: 'Action: calculator(1e300 ** 2)' crashed the run after one call
    client = scripted("Action: calculator(1e300 ** 2)", "Final Answer: too large")
    assert ra.run("huge") == "too large"
    assert "Error: result is too large" in client.requests[1]["messages"][-1]["content"]


def test_inline_final_answer_without_action(scripted):
    scripted("Thought: easy. Final Answer: 4")
    assert ra.run("2+2") == "4"
