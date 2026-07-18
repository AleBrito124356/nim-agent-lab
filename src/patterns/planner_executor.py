"""Planner-executor with replanning.

Three roles, one loop:

1. PLANNER decomposes the goal into a short ordered list of steps.
2. EXECUTOR runs the current step (it can request the calculator tool).
3. REPLANNER looks at the result and revises the REMAINING steps -- steps
   that became unnecessary get dropped, missing ones get inserted.

This separation matters in practice: a single-prompt agent commits to its
first plan even when step 2 proves it wrong. The replanner is what makes the
pattern robust to surprises.

Run standalone:
    python -m src.patterns.planner_executor "Plan a 3-day developer conference budget for 120 people at 85 USD per head per day"
"""

from __future__ import annotations

import ast
import json
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

DESCRIPTION = "Planner decomposes the goal, executor runs each step, replanner revises after every result."
DEFAULT_GOAL = (
    "Estimate the total cost of a 3-day workshop for 25 people: venue is 400 "
    "USD per day, catering is 18 USD per person per day, and each attendee "
    "gets a 35 USD materials kit. Then summarize the budget in three lines."
)

MAX_STEPS = 8

# --- minimal safe calculator the executor can call --------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError) as exc:
        return f"Error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- JSON plan parsing ------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (handles ``` fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_steps(reply: str) -> list[str]:
    data = _extract_json(reply)
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        steps = [str(s).strip() for s in data["steps"] if str(s).strip()]
        return steps[:MAX_STEPS]
    return []


# --- roles ------------------------------------------------------------------

PLANNER_PROMPT = """\
You are a planner. Decompose the user's goal into the SMALLEST ordered list
of concrete steps (2 to 6). Each step must be a single self-contained
instruction an executor can carry out with plain reasoning plus a calculator.

Respond with ONLY this JSON, no prose:
{"steps": ["step 1", "step 2", ...]}
"""

EXECUTOR_PROMPT = """\
You are an executor working through a plan one step at a time.
If the current step needs arithmetic, first reply with exactly one line:
CALC: <arithmetic expression>
and wait for the result. Otherwise (or after receiving calculator results)
reply with the step's outcome as concise prose. Never invent numbers.
"""

REPLANNER_PROMPT = """\
You are a replanner. Given the goal, the completed steps with their results,
and the remaining planned steps, revise the REMAINING steps only:
- drop steps whose work is already done,
- add a step if something essential is missing,
- keep it minimal.

Respond with ONLY this JSON, no prose:
{"steps": ["remaining step 1", ...]}
If nothing remains, respond {"steps": []}.
"""


def _execute_step(client, goal: str, step: str, completed: list[tuple[str, str]]) -> str:
    """Run one step; grant the executor up to 3 calculator calls."""
    history = "\n".join(f"- {s}\n  result: {r}" for s, r in completed) or "(none yet)"
    messages = [
        {"role": "system", "content": EXECUTOR_PROMPT},
        {
            "role": "user",
            "content": (
                f"Overall goal: {goal}\n\nCompleted steps:\n{history}\n\n"
                f"Current step: {step}"
            ),
        },
    ]
    for _ in range(3):
        reply = chat(client, messages, temperature=0.0)
        calc_match = re.match(r"^\s*CALC:\s*(.+)$", reply, re.MULTILINE)
        if not calc_match:
            return reply.strip()
        expr = calc_match.group(1).strip()
        result = calculator(expr)
        print(f"    [calc] {expr} = {result}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Calculator result: {result}"})
    return chat(client, messages, temperature=0.0).strip()


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[planner-executor] goal: {goal}\n" + "-" * 72)

    plan_reply = chat(
        client,
        [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": goal},
        ],
        temperature=0.0,
    )
    plan = _parse_steps(plan_reply)
    if not plan:
        plan = [f"Answer the goal directly: {goal}"]
    print("[plan]")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    completed: list[tuple[str, str]] = []
    executed = 0
    while plan and executed < MAX_STEPS:
        step = plan.pop(0)
        executed += 1
        print(f"\n[execute {executed}] {step}")
        result = _execute_step(client, goal, step, completed)
        print(f"  -> {result}")
        completed.append((step, result))

        if not plan:
            break

        history = "\n".join(f"- {s}\n  result: {r}" for s, r in completed)
        remaining = "\n".join(f"- {s}" for s in plan)
        replan_reply = chat(
            client,
            [
                {"role": "system", "content": REPLANNER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n\nCompleted:\n{history}\n\n"
                        f"Remaining plan:\n{remaining}"
                    ),
                },
            ],
            temperature=0.0,
        )
        revised = _parse_steps(replan_reply)
        if revised != plan and _extract_json(replan_reply) is not None:
            print("[replan] remaining steps revised:")
            for i, step_text in enumerate(revised, 1):
                print(f"  {i}. {step_text}")
            plan = revised

    summary_input = "\n".join(f"- {s}\n  result: {r}" for s, r in completed)
    final = chat(
        client,
        [
            {
                "role": "system",
                "content": "Merge the step results into one direct, complete answer to the goal.",
            },
            {"role": "user", "content": f"Goal: {goal}\n\nStep results:\n{summary_input}"},
        ],
        temperature=0.2,
    )
    print("-" * 72 + f"\n[planner-executor] finished after {executed} step(s).")
    return final


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nFINAL ANSWER:\n{final}")
