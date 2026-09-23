"""Planner-executor with replanning.

Three roles, one loop:

1. PLANNER decomposes the goal into a short ordered list of steps.
2. EXECUTOR runs the current step (it can request the calculator tool).
3. REPLANNER looks at the result and revises the REMAINING steps -- steps
   that became unnecessary get dropped, missing ones get inserted.

This separation matters in practice: a single-prompt agent commits to its
first plan even when step 2 proves it wrong. The replanner is what makes the
pattern robust to surprises.

The replanner is also a failure point: if its reply has no usable steps list
(prose, or JSON under an unexpected key) the remaining plan is KEPT. Treating
"unparseable" as "nothing left to do" would silently skip the rest of the work.

Run standalone:
    python -m src.patterns.planner_executor "Plan a 3-day developer conference budget for 120 people at 85 USD per head per day"
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, extract_json, get_client  # noqa: E402

DESCRIPTION = "Planner decomposes the goal, executor runs each step, replanner revises after every result."
DEFAULT_GOAL = (
    "Estimate the total cost of a 3-day workshop for 25 people: venue is 400 "
    "USD per day, catering is 18 USD per person per day, and each attendee "
    "gets a 35 USD materials kit. Then summarize the budget in three lines."
)

MAX_STEPS = 8
MAX_CALCS_PER_STEP = 3

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
MAX_EXPRESSION_CHARS = 500
MAX_RESULT_BITS = 3322  # ~1000 decimal digits: far below Python's int->str limit


def _checked(value):
    """Reject results a tool must never hand back: complex, inf/nan, giant ints."""
    if isinstance(value, complex):
        raise ValueError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is too large to represent")
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise ValueError("result is too large (over ~1000 digits)")
    return value


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("exponent too large (limit: 100)")
            if isinstance(left, int) and isinstance(right, int) and left.bit_length() * right > MAX_RESULT_BITS:
                raise ValueError("result is too large (over ~1000 digits)")
        return _checked(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _checked(_UNARY_OPS[type(node.op)](_safe_eval(node.operand)))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    expression = str(expression).strip()
    if len(expression) > MAX_EXPRESSION_CHARS:
        return f"Error: expression longer than {MAX_EXPRESSION_CHARS} characters."
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except OverflowError:
        return "Error: result is too large to represent."
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError) as exc:
        return f"Error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- JSON plan parsing ------------------------------------------------------

# Keys a model may use for the step list, and for the text inside a dict step.
_STEP_LIST_KEYS = ("steps", "remaining_steps", "plan")
_STEP_TEXT_KEYS = ("description", "instruction", "task", "action", "step", "text", "title")


def _step_text(item) -> str:
    """Render one plan item as an instruction: strings pass through, dict
    steps like {"step": 1, "action": "add"} become "add"."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in _STEP_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return " ".join(str(v).strip() for v in item.values() if isinstance(v, str) and v.strip())
    return ""  # numbers, nulls and nested lists carry no instruction


def _parse_steps(reply: str) -> list[str] | None:
    """Return the step list, [] for an explicitly empty plan, or None when the
    reply contains no usable steps list at all."""
    data = extract_json(reply)
    if data is None:
        return None
    for key in _STEP_LIST_KEYS:
        if isinstance(data.get(key), list):
            steps = [text for text in (_step_text(s) for s in data[key]) if text]
            return steps[:MAX_STEPS]
    return None


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

_CALC_RE = re.compile(r"^\s*CALC:\s*(.+?)\s*$", re.MULTILINE)


def _execute_step(client, goal: str, step: str, completed: list[tuple[str, str]]) -> str:
    """Run one step; grant the executor up to MAX_CALCS_PER_STEP calculator calls."""
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
    for _ in range(MAX_CALCS_PER_STEP):
        reply = chat(client, messages, temperature=0.0)
        calc_match = _CALC_RE.search(reply)  # models often add a sentence first
        if not calc_match:
            return reply.strip()
        expr = calc_match.group(1).strip()
        result = calculator(expr)
        print(f"    [calc] {expr} = {result}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Calculator result: {result}"})
    messages.append(
        {
            "role": "user",
            "content": "Calculator budget for this step is used up. Reply now with the "
            "step's outcome in prose, using the results above. No more CALC lines.",
        }
    )
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
        print("[plan] planner reply had no usable steps; answering the goal in one step.")
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
        if revised is None:
            print("[replan] reply had no steps list; keeping the remaining plan.")
        elif revised != plan:
            print("[replan] remaining steps revised:")
            for i, step_text in enumerate(revised, 1):
                print(f"  {i}. {step_text}")
            if not revised:
                print("  (none -- the replanner judged the goal complete)")
            plan = revised

    if plan:
        print(f"[planner-executor] step budget ({MAX_STEPS}) reached; {len(plan)} step(s) left undone.")

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
