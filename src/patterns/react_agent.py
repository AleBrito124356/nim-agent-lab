"""ReAct agent: interleaved Thought / Action / Observation reasoning.

The model reasons in plain text, requests tools with a strict
``Action: tool(args)`` line, and receives tool output back as an
``Observation:`` message. The loop ends when the model emits
``Final Answer:`` or the iteration budget runs out.

All three tools are implemented locally in this file so the pattern is fully
self-contained: a safe calculator (AST walk, no eval), a unit converter, and
a deterministic mock weather lookup.

Text parsing is where ReAct agents break in practice, so the parser here is
deliberately forgiving about what models really emit: quoted or keyword
arguments (``convert_units(42, "km", "mi")``), nested parentheses, trailing
prose after the call, the LangChain-style ``Action Input:`` line, and
self-written ``Observation:`` lines (which are discarded -- only the runtime
may supply observations).

Run standalone:
    python -m src.patterns.react_agent "How many miles is 42 km, and what is that value squared?"
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

DESCRIPTION = "Thought -> Action -> Observation loop with a calculator, unit converter and weather tool."
DEFAULT_GOAL = (
    "Convert 42 kilometers to miles, square the result, then tell me the "
    "current weather in Panama City."
)

MAX_ITERATIONS = 8

# --------------------------------------------------------------------------
# Tools
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


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST. Anything non-arithmetic raises."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("exponent too large (limit: 100)")
            # Refuse before computing: (9**99)**99 would build a 9,000-digit int.
            if isinstance(left, int) and isinstance(right, int) and left.bit_length() * right > MAX_RESULT_BITS:
                raise ValueError("result is too large (over ~1000 digits)")
        return _checked(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _checked(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression without eval(): +, -, *, /, //, %, **.

    Every failure -- bad syntax, division by zero, overflow, absurdly large
    results -- comes back as an "Error: ..." string the model can read and
    recover from. A tool must never crash the agent loop.
    """
    expression = str(expression).strip()
    if len(expression) > MAX_EXPRESSION_CHARS:
        return f"Error: expression longer than {MAX_EXPRESSION_CHARS} characters."
    try:
        result = _eval_node(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except OverflowError:
        return "Error: result is too large to represent."
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError) as exc:
        return f"Error: could not evaluate {expression!r} ({exc})."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# Unit -> (dimension, factor to base unit). Base units: meter, kilogram, liter.
_LINEAR_UNITS = {
    "km": ("length", 1000.0),
    "m": ("length", 1.0),
    "mi": ("length", 1609.344),
    "ft": ("length", 0.3048),
    "kg": ("mass", 1.0),
    "lb": ("mass", 0.45359237),
    "l": ("volume", 1.0),
    "gal": ("volume", 3.785411784),
}


def convert_units(value_str: str, from_unit: str, to_unit: str) -> str:
    """Convert between km/m/mi/ft, kg/lb, l/gal, and c/f temperatures."""
    try:
        value = float(str(value_str).strip())
    except ValueError:
        return f"Error: {value_str!r} is not a number."
    if not math.isfinite(value):
        return f"Error: {value_str!r} is not a finite number."
    src, dst = str(from_unit).strip().lower(), str(to_unit).strip().lower()

    if {src, dst} <= {"c", "f"}:
        if src == dst:
            result = value
        elif src == "c":
            result = value * 9 / 5 + 32
        else:
            result = (value - 32) * 5 / 9
        return f"{result:.4g} {dst}"

    if src not in _LINEAR_UNITS or dst not in _LINEAR_UNITS:
        supported = ", ".join(sorted(_LINEAR_UNITS) + ["c", "f"])
        return f"Error: unsupported unit. Supported: {supported}."
    src_dim, src_factor = _LINEAR_UNITS[src]
    dst_dim, dst_factor = _LINEAR_UNITS[dst]
    if src_dim != dst_dim:
        return f"Error: cannot convert {src} ({src_dim}) to {dst} ({dst_dim})."
    result = value * src_factor / dst_factor
    return f"{result:.6g} {dst}"


_CONDITIONS = ["sunny", "partly cloudy", "overcast", "light rain", "thunderstorms"]


def get_weather(city: str) -> str:
    """Deterministic mock weather: same city always returns the same report.

    Real deployments swap this for an HTTP call; the point of the demo is the
    loop mechanics, not a weather API key.
    """
    city = str(city).strip()
    if not city:
        return "Error: get_weather needs a city name, e.g. get_weather(Panama City)."
    seed = zlib.crc32(city.lower().encode("utf-8"))
    temp_c = 18 + seed % 15
    condition = _CONDITIONS[seed % len(_CONDITIONS)]
    humidity = 40 + seed % 55
    return f"{city}: {temp_c} C, {condition}, humidity {humidity}% (demo data)"


# --------------------------------------------------------------------------
# Parsing the model's Action line
# --------------------------------------------------------------------------

_ACTION_HEAD_RE = re.compile(r"^[ \t]*Action:[ \t]*([A-Za-z_]\w*)[ \t]*(\(?)", re.MULTILINE)
_ACTION_INPUT_RE = re.compile(r"^[ \t]*Action Input:[ \t]*(.*)$", re.MULTILINE)
_OBSERVATION_RE = re.compile(r"^[ \t]*Observation:", re.MULTILINE)
_FINAL_LINE_RE = re.compile(r"^[ \t]*Final Answer:", re.MULTILINE)
_KWARG_RE = re.compile(r"^[A-Za-z_]\w*\s*=(?!=)\s*")
_QUOTES = "\"'"


def _balanced_args(text: str, start: int) -> str:
    """Return the text between the '(' that ends at ``start`` and its matching
    ')' on the same line. Nested parentheses and quoted commas are respected;
    anything after the closing parenthesis is ignored."""
    depth, quote = 1, None
    line_end = text.find("\n", start)
    line_end = len(text) if line_end == -1 else line_end
    for i in range(start, line_end):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in _QUOTES:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
    # Unbalanced (or an apostrophe opened a "quote"): take the rest of the line.
    rest = text[start:line_end].rstrip()
    return rest[:-1] if rest.endswith(")") else rest


def parse_action(reply: str) -> tuple[str, str] | None:
    """Extract (tool, raw_args) from the first Action line, or None.

    Accepts ``Action: tool(args)`` and the LangChain-style pair
    ``Action: tool`` + ``Action Input: args`` that many models were trained on.
    The match is bounded to one line, so a later "Thought: ... calculator(x)"
    can never leak into the arguments.
    """
    match = _ACTION_HEAD_RE.search(reply)
    if not match:
        return None
    tool = match.group(1)
    if match.group(2):
        return tool, _balanced_args(reply, match.end())
    action_input = _ACTION_INPUT_RE.search(reply, match.end())
    return tool, action_input.group(1).strip() if action_input else ""


def _clean_arg(arg: str) -> str:
    """Strip whitespace, a keyword prefix (value=42) and matching quotes."""
    arg = _KWARG_RE.sub("", arg.strip(), count=1).strip()
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in _QUOTES:
        arg = arg[1:-1].strip()
    return arg


def split_args(raw_args: str) -> list[str]:
    """Split on commas outside quotes/brackets, then clean each argument, so
    convert_units(42, "km", 'mi') and convert_units(value=42, ...) both work."""
    parts, buf, quote, depth = [], [], None, 0
    for ch in raw_args:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in _QUOTES:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [_clean_arg(part) for part in parts]


def _dispatch(tool: str, raw_args: str) -> str:
    """Route a parsed Action to the matching local tool."""
    if tool == "calculator":
        return calculator(_clean_arg(raw_args))
    if tool == "convert_units":
        parts = split_args(raw_args)
        if len(parts) != 3:
            return "Error: convert_units expects (value, from_unit, to_unit)."
        return convert_units(*parts)
    if tool == "get_weather":
        return get_weather(_clean_arg(raw_args))
    return f"Error: unknown tool {tool!r}. Use calculator, convert_units or get_weather."


def _truncate_observation(reply: str) -> str:
    """Drop any Observation the model wrote itself (some models ignore stop=)."""
    match = _OBSERVATION_RE.search(reply)
    return reply[: match.start()].rstrip() if match else reply


def _final_answer(reply: str, action: tuple[str, str] | None) -> str | None:
    """Return the Final Answer text, unless an Action comes first.

    A Final Answer written AFTER an Action line was guessed before the tool
    ran; in that case the tool runs and the model answers from the real
    observation on its next turn.
    """
    match = _FINAL_LINE_RE.search(reply)
    if match is None and action is None:
        match = re.search(r"Final Answer:", reply)  # inline, when no Action exists
    if match is None:
        return None
    if action is not None and reply.find("Action:") < match.start():
        return None
    return reply[match.end():].strip()


# --------------------------------------------------------------------------
# ReAct loop
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a ReAct agent. Solve the task step by step, using tools when needed.

Available tools:
- calculator(expression) -> evaluate arithmetic. Example: calculator(23 * 7 + 1)
- convert_units(value, from_unit, to_unit) -> supported units: km, m, mi, ft, kg, lb, l, gal, c, f. Example: convert_units(10, km, mi)
- get_weather(city) -> current weather for a city. Example: get_weather(Panama City)

Reply with EXACTLY ONE step per turn, in one of these two forms:

Thought: <what you need to figure out next>
Action: <tool_name>(<args>)

or, when you can answer:

Thought: <why you are done>
Final Answer: <the complete answer for the user>

Never write an Observation yourself. The system provides it after each Action.
"""


def run(goal: str) -> str:
    """Run the ReAct loop until Final Answer or MAX_ITERATIONS."""
    client = get_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {goal}"},
    ]
    print(f"\n[react] goal: {goal}\n" + "-" * 72)

    for step in range(1, MAX_ITERATIONS + 1):
        # stop= prevents the model from hallucinating its own Observation;
        # _truncate_observation covers models that ignore stop sequences.
        reply = _truncate_observation(
            chat(client, messages, temperature=0.0, stop=["Observation:"])
        )
        print(f"\n[step {step}]\n{reply}")
        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        answer = _final_answer(reply, action)
        if answer is not None:
            print("-" * 72 + f"\n[react] done in {step} step(s).")
            return answer

        if action is None:
            observation = (
                "Error: reply had no valid 'Action: tool(args)' line and no "
                "'Final Answer:'. Follow the format exactly."
            )
        else:
            observation = _dispatch(*action)
        print(f"Observation: {observation}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    print("-" * 72 + "\n[react] iteration budget exhausted.")
    return "The agent hit its iteration limit before producing a final answer."


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nFINAL ANSWER: {final}")
