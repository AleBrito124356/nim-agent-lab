"""The four self-contained calculator copies must agree, and must never raise.

Regression for the audit finding: 1e300**2 (OverflowError) and (9**99)**99
(int->str digit limit) used to escape every copy and crash the agent loop.
"""

from __future__ import annotations

import json

import pytest

from src.patterns import human_in_the_loop, planner_executor, react_agent, tool_calling_agent

CALCULATORS = {
    "react": react_agent.calculator,
    "tool-calling": tool_calling_agent.calculator,
    "planner-executor": planner_executor.calculator,
    "human-in-the-loop": human_in_the_loop.tool_calculator,
}


def outcome(calc, expression):
    """Normalize the two output styles to ("ok", number) or ("error", text)."""
    out = calc(expression)
    assert isinstance(out, str)
    if out.startswith("{"):
        data = json.loads(out)
        return ("error", data["error"]) if "error" in data else ("ok", float(data["result"]))
    if out.startswith("Error"):
        return "error", out
    return "ok", float(out)


VALID = [
    ("2 + 3 * 4", 14),
    ("(2 + 3) * 4", 20),
    ("7 / 2", 3.5),
    ("7 // 2", 3),
    ("7 % 4", 3),
    ("2 ** 10", 1024),
    ("-3 + +5", 2),
    ("26.0976 ** 2", 681.08472576),
    ("40 * 52 * 0.2", 416),
    ("10 ** 100", 1e100),
]

INVALID = [
    "1 / 0",
    "1e300 ** 2",                        # float overflow -> OverflowError
    "(9 ** 99) ** 99",                   # 9,000+ digit int -> int->str limit
    "10 ** 100 * " * 10 + "10 ** 100",   # product over ~1000 digits
    "2 ** 1000",                         # exponent limit
    "1e308 * 10",                        # inf without an exception
    "(-8) ** 0.5",                       # complex result
    "__import__('os').system('echo hi')",
    "abs(-1)",
    "x + 1",
    "True + 1",
    "'a' * 3",
    "",
    "1 +",
    "1" * 600,
    "(" * 250 + "1" + ")" * 250,
]


@pytest.mark.parametrize("name", CALCULATORS)
@pytest.mark.parametrize("expression,expected", VALID)
def test_valid_expressions(name, expression, expected):
    status, value = outcome(CALCULATORS[name], expression)
    assert status == "ok"
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("name", CALCULATORS)
@pytest.mark.parametrize("expression", INVALID)
def test_invalid_expressions_return_errors_instead_of_raising(name, expression):
    status, _ = outcome(CALCULATORS[name], expression)
    assert status == "error"


@pytest.mark.parametrize("name", CALCULATORS)
def test_integer_results_print_without_decimal(name):
    out = CALCULATORS[name]("6 * 7")
    assert "42" in out and "42.0" not in out


@pytest.mark.parametrize("name", CALCULATORS)
def test_non_string_input_is_coerced(name):
    status, value = outcome(CALCULATORS[name], 12)
    assert (status, value) == ("ok", 12)
