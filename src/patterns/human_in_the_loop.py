"""Human-in-the-loop: the agent proposes, the human disposes.

The agent plans a set of tool actions but executes NOTHING until each action
is individually approved at the terminal. The reviewer can approve, edit the
arguments before execution, or reject. Every decision -- including edits and
rejections -- is appended to a JSONL audit log on disk, which is the piece
compliance teams actually ask for.

Proposals the runtime cannot execute (an unknown tool, arguments that are
not an object, a bare string instead of an action) are never shown for
approval, but they ARE written to the audit log as "discarded" with the
reason: an audit trail that only records the well-formed requests hides
exactly the behaviour a reviewer needs to see.

When stdin is not a terminal (CI, piped input, offline replay under --all)
the pattern degrades to a dry run: proposals are logged as auto-skipped and
nothing executes.

Run standalone:
    python -m src.patterns.human_in_the_loop "Prepare a note summarizing Q3 priorities and email it to the team"
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, extract_json, get_client  # noqa: E402

DESCRIPTION = "Agent proposes actions; each needs terminal approval; every decision is audit-logged."
DEFAULT_GOAL = (
    "Create a note called 'q3-priorities' listing three realistic Q3 "
    "priorities for a two-person SaaS team, compute 40 * 52 * 0.2 as the "
    "yearly maintenance hours estimate, and email the note to the team."
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "hitl_workspace"
AUDIT_LOG = REPO_ROOT / "audit_log.jsonl"
MAX_ACTIONS = 6

# --------------------------------------------------------------------------
# Tools the agent may propose
# --------------------------------------------------------------------------

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


def tool_calculator(expression: str) -> str:
    expression = str(expression).strip()
    if len(expression) > MAX_EXPRESSION_CHARS:
        return f"Error: expression longer than {MAX_EXPRESSION_CHARS} characters."
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except OverflowError:
        return "Error: result is too large to represent."
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, RecursionError, MemoryError) as exc:
        return f"Error: {exc or type(exc).__name__}"
    return str(int(result) if isinstance(result, float) and result.is_integer() else result)


def tool_write_note(name: str, content: str) -> str:
    """Write a text note inside the workspace directory only."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", str(name))[:60]
    if not safe_name:
        return "Error: note name reduced to nothing after sanitization."
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE / f"{safe_name}.txt"
    path.write_text(str(content), encoding="utf-8")
    return f"Wrote {len(str(content))} chars to {path}"


def tool_send_email(to: str, subject: str, body: str) -> str:
    """SIMULATED email send -- prints instead of sending. Swap for SMTP in prod."""
    print("\n  --- simulated email ---")
    print(f"  To: {to}\n  Subject: {subject}\n  Body:\n  {str(body)[:400]}")
    print("  --- end (nothing was actually sent) ---")
    return f"Simulated send to {to} with subject {subject!r}."


TOOLS = {
    "calculator": (tool_calculator, ["expression"]),
    "write_note": (tool_write_note, ["name", "content"]),
    "send_email": (tool_send_email, ["to", "subject", "body"]),
}

PLANNER_PROMPT = """\
You are an assistant that PROPOSES actions for human review; you never
execute anything yourself. Available tools:
- calculator(expression)
- write_note(name, content)          # name: alphanumeric/dash/underscore
- send_email(to, subject, body)

Plan the minimal set of actions (max 6) for the user's goal.
Respond with ONLY this JSON:
{"actions": [{"tool": "<tool>", "args": {"<param>": "<value>"}, "reason": "<why>"}]}
"""


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------


def audit(entry: dict) -> None:
    entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------
# Review loop
# --------------------------------------------------------------------------


def _normalize_action(item) -> tuple[dict | None, str]:
    """Validate one proposal. Returns (action, "") or (None, why-discarded).

    Tolerates the common drift: "name"/"action" instead of "tool",
    "arguments"/"parameters" instead of "args", args sent as a JSON string.
    """
    if not isinstance(item, dict):
        return None, f"not an action object: {str(item)[:80]!r}"
    tool = str(item.get("tool") or item.get("name") or item.get("action") or "").strip()
    if tool not in TOOLS:
        return None, f"unknown tool {tool!r}"
    args = next(
        (item[key] for key in ("args", "arguments", "parameters") if key in item), None
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    if not isinstance(args, dict):
        return None, f"args for {tool} must be an object, got {type(args).__name__}"
    return {"tool": tool, "args": args, "reason": str(item.get("reason", ""))}, ""


def _parse_actions(reply: str) -> tuple[list[dict], list[dict]]:
    """Return (valid actions, discarded proposals with reasons)."""
    data = extract_json(reply)
    items = data.get("actions") if data is not None else None
    if not isinstance(items, list):
        return [], []
    actions, discarded = [], []
    for position, item in enumerate(items):
        if position >= MAX_ACTIONS:
            discarded.append({"proposal": item, "why": f"over the {MAX_ACTIONS}-action limit"})
            continue
        action, why = _normalize_action(item)
        if action is None:
            discarded.append({"proposal": item, "why": why})
        else:
            actions.append(action)
    return actions, discarded


def _ask(prompt: str) -> str | None:
    """input() that reports a closed stdin as None instead of raising.

    isatty() is not enough on its own: on Windows the NUL device (what
    `< /dev/null` maps to) claims to be a terminal, then hits EOF at once.
    """
    try:
        return input(prompt)
    except EOFError:
        print()
        return None


def _review(action: dict) -> tuple[str, dict]:
    """Ask the human. Returns (decision, possibly-edited args)."""
    if not sys.stdin.isatty():
        return "auto-skipped (non-interactive)", action["args"]
    while True:
        choice = _ask("  [a]pprove / [e]dit args / [r]eject > ")
        if choice is None:
            return "auto-skipped (reviewer input closed)", action["args"]
        choice = choice.strip().lower()
        if choice in {"a", "approve"}:
            return "approved", action["args"]
        if choice in {"r", "reject"}:
            return "rejected", action["args"]
        if choice in {"e", "edit"}:
            print(f"  current args: {json.dumps(action['args'])}")
            raw = _ask("  new args as JSON > ")
            if raw is None:
                return "auto-skipped (reviewer input closed)", action["args"]
            raw = raw.strip()
            try:
                new_args = json.loads(raw)
                if isinstance(new_args, dict):
                    return "approved-with-edits", new_args
                print("  must be a JSON object.")
            except json.JSONDecodeError as exc:
                print(f"  invalid JSON ({exc}); try again.")
        else:
            print("  type a, e or r.")


def _execute(tool_name: str, args: dict) -> str:
    fn, params = TOOLS[tool_name]
    missing = [p for p in params if p not in args]
    if missing:
        return f"Error: missing argument(s): {missing}"
    try:
        return fn(**{p: args[p] for p in params})
    except Exception as exc:  # tool errors must not kill the review session
        return f"Error: {exc}"


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[hitl] goal: {goal}")
    print(f"[hitl] audit log: {AUDIT_LOG}\n" + "-" * 72)

    reply = chat(
        client,
        [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": goal},
        ],
        temperature=0.2,
    )
    actions, discarded = _parse_actions(reply)

    for item in discarded:
        print(f"\n[discarded] {item['why']} -- not offered for approval")
        audit({"goal": goal, "decision": "discarded", "why": item["why"],
               "proposal": item["proposal"], "result": None})

    if not actions:
        return "The agent proposed no valid actions."

    results: list[str] = []
    for i, action in enumerate(actions, 1):
        print(f"\n[proposal {i}/{len(actions)}] {action['tool']}")
        print(f"  args:   {json.dumps(action['args'], ensure_ascii=False)}")
        print(f"  reason: {action['reason']}")

        decision, final_args = _review(action)
        entry = {
            "goal": goal,
            "tool": action["tool"],
            "proposed_args": action["args"],
            "final_args": final_args,
            "decision": decision,
        }
        if decision.startswith("approved"):
            result = _execute(action["tool"], final_args)
            print(f"  executed -> {result}")
            entry["result"] = result
            results.append(f"{action['tool']}: {result}")
        else:
            print(f"  {decision}; not executed.")
            entry["result"] = None
            results.append(f"{action['tool']}: {decision}")
        audit(entry)

    results.extend(f"discarded proposal: {item['why']}" for item in discarded)
    summary = "\n".join(f"- {r}" for r in results)
    logged = len(actions) + len(discarded)
    print("-" * 72 + f"\n[hitl] session complete; {logged} decision(s) audit-logged.")
    return summary


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nSESSION SUMMARY:\n{final}")
