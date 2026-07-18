"""Human-in-the-loop: the agent proposes, the human disposes.

The agent plans a set of tool actions but executes NOTHING until each action
is individually approved at the terminal. The reviewer can approve, edit the
arguments before execution, or reject. Every decision -- including edits and
rejections -- is appended to a JSONL audit log on disk, which is the piece
compliance teams actually ask for.

When stdin is not a terminal (CI, piped input) the pattern degrades to a
dry run: proposals are logged as auto-skipped and nothing executes.

Run standalone:
    python -m src.patterns.human_in_the_loop "Prepare a note summarizing Q3 priorities and email it to the team"
"""

from __future__ import annotations

import ast
import json
import operator
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

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


def tool_calculator(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(str(expression).strip(), mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: {exc}"
    return str(int(result) if isinstance(result, float) and result.is_integer() else result)


def tool_write_note(name: str, content: str) -> str:
    """Write a text note inside the workspace directory only."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", str(name))[:60]
    if not safe_name:
        return "Error: note name reduced to nothing after sanitization."
    WORKSPACE.mkdir(exist_ok=True)
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
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Review loop
# --------------------------------------------------------------------------


def _parse_actions(reply: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", reply).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    actions = []
    for item in data.get("actions", [])[:MAX_ACTIONS]:
        tool = str(item.get("tool", ""))
        if tool in TOOLS and isinstance(item.get("args"), dict):
            actions.append(
                {"tool": tool, "args": item["args"], "reason": str(item.get("reason", ""))}
            )
    return actions


def _review(action: dict) -> tuple[str, dict]:
    """Ask the human. Returns (decision, possibly-edited args)."""
    if not sys.stdin.isatty():
        return "auto-skipped (non-interactive)", action["args"]
    while True:
        choice = input("  [a]pprove / [e]dit args / [r]eject > ").strip().lower()
        if choice in {"a", "approve"}:
            return "approved", action["args"]
        if choice in {"r", "reject"}:
            return "rejected", action["args"]
        if choice in {"e", "edit"}:
            print(f"  current args: {json.dumps(action['args'])}")
            raw = input("  new args as JSON > ").strip()
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
    actions = _parse_actions(reply)
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

    summary = "\n".join(f"- {r}" for r in results)
    print("-" * 72 + f"\n[hitl] session complete; {len(actions)} decision(s) audit-logged.")
    return summary


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nSESSION SUMMARY:\n{final}")
