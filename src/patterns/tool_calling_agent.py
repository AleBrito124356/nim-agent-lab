"""Native function calling: the model emits structured tool_calls, not text.

Compared to the text-parsed ReAct loop, native tool calling is more robust
(no regex parsing of model output) and supports PARALLEL tool calls -- the
model can request several independent lookups in a single turn, and this
loop executes every one of them before replying.

Run standalone:
    python -m src.patterns.tool_calling_agent "What is 250 USD in EUR and JPY, and what time is it in UTC-5?"
"""

from __future__ import annotations

import ast
import json
import operator
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import get_client, get_model  # noqa: E402

DESCRIPTION = "OpenAI-style function calling with JSON-schema tools and parallel call handling."
DEFAULT_GOAL = (
    "Convert 250 USD to EUR and to JPY, then compute the total of the two "
    "converted amounts times 1.07, and tell me the current time in UTC-5."
)

MAX_ROUNDS = 6

# --------------------------------------------------------------------------
# Tool implementations (local, deterministic where possible)
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


def calculator(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return json.dumps({"error": "division by zero"})
    except (ValueError, SyntaxError) as exc:
        return json.dumps({"error": f"could not evaluate: {exc}"})
    return json.dumps({"result": result})


# Fixed demo table so runs are reproducible. Swap for a real FX API in prod.
_USD_RATES = {"USD": 1.0, "EUR": 0.92, "JPY": 155.7, "GBP": 0.79, "PAB": 1.0, "MXN": 17.1}


def get_exchange_rate(base: str, quote: str) -> str:
    base, quote = base.upper().strip(), quote.upper().strip()
    if base not in _USD_RATES or quote not in _USD_RATES:
        return json.dumps(
            {"error": f"unsupported currency; supported: {sorted(_USD_RATES)}"}
        )
    rate = _USD_RATES[quote] / _USD_RATES[base]
    return json.dumps({"base": base, "quote": quote, "rate": round(rate, 6), "source": "demo table"})


def get_current_time(utc_offset_hours: float = 0.0) -> str:
    if not -14 <= utc_offset_hours <= 14:
        return json.dumps({"error": "utc_offset_hours must be between -14 and 14"})
    tz = timezone(timedelta(hours=utc_offset_hours))
    now = datetime.now(tz)
    return json.dumps(
        {"iso": now.isoformat(timespec="seconds"), "utc_offset_hours": utc_offset_hours}
    )


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_exchange_rate": get_exchange_rate,
    "get_current_time": get_current_time,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression (+, -, *, /, //, %, **).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(230 + 38935) * 1.07'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate between two currencies (USD, EUR, JPY, GBP, PAB, MXN).",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Currency to convert from, e.g. 'USD'"},
                    "quote": {"type": "string", "description": "Currency to convert to, e.g. 'EUR'"},
                },
                "required": ["base", "quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time at a given UTC offset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_offset_hours": {
                        "type": "number",
                        "description": "Hours relative to UTC, e.g. -5 for Panama/EST",
                    }
                },
                "required": ["utc_offset_hours"],
            },
        },
    },
]

# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------


def _execute_tool_call(tool_call) -> str:
    """Execute one tool call; always returns a JSON string for the tool message."""
    name = tool_call.function.name
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name!r}"})
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"malformed arguments: {exc}"})
    try:
        return fn(**args)
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})


def run(goal: str) -> str:
    """Loop: model -> tool_calls -> execute all -> feed results back -> repeat."""
    client = get_client()
    model = get_model()
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Use the provided tools for any "
                "math, currency or time question -- never guess numbers. "
                "Request independent lookups in parallel when possible."
            ),
        },
        {"role": "user", "content": goal},
    ]
    print(f"\n[tool-calling] goal: {goal}\n" + "-" * 72)

    for round_no in range(1, MAX_ROUNDS + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1024,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            print(f"\n[round {round_no}] final answer ready.")
            return answer

        # Serialize the assistant turn (content may be None during tool use).
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # Parallel handling: execute EVERY call the model requested this turn.
        print(f"\n[round {round_no}] model requested {len(msg.tool_calls)} tool call(s):")
        for tc in msg.tool_calls:
            result = _execute_tool_call(tc)
            print(f"  {tc.function.name}({tc.function.arguments}) -> {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

    return "The agent hit its round limit before producing a final answer."


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print("-" * 72 + f"\nFINAL ANSWER: {final}")
