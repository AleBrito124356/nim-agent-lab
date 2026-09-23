"""Router: classify the request, dispatch to a specialist persona.

One generalist prompt that does everything does nothing well. This pattern
runs a cheap classification pass first, then hands the request to a
specialist with its own system prompt AND its own sampling temperature --
the coder runs cold (0.1), the writer runs warm (0.8). That per-route
temperature is the detail most router tutorials skip, and it is half the
value of the pattern.

Run standalone:
    python -m src.patterns.router_agent "Write a Python function that merges two sorted lists in O(n)"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, extract_json, get_client  # noqa: E402

DESCRIPTION = "Classifies each request and routes it to a coder, writer or analyst persona."
DEFAULT_GOAL = (
    "Our API error rate went from 0.2% to 1.1% after Tuesday's deploy. "
    "What should we look at first?"
)

SPECIALISTS: dict[str, dict] = {
    "coder": {
        "temperature": 0.1,
        "system": (
            "You are a senior software engineer. Give working, idiomatic code "
            "with brief comments where intent is not obvious. State complexity "
            "and edge cases. No filler prose around the code."
        ),
    },
    "writer": {
        "temperature": 0.8,
        "system": (
            "You are a professional writer. Produce clear, engaging prose "
            "matched to the requested format and audience. Vary sentence "
            "rhythm; cut every word that does no work."
        ),
    },
    "analyst": {
        "temperature": 0.3,
        "system": (
            "You are a pragmatic technical analyst. Structure answers as: "
            "likely causes ranked by probability, how to confirm each, and "
            "what to do about it. Be explicit about uncertainty."
        ),
    },
}

CLASSIFIER_PROMPT = f"""\
Classify the user request into exactly one route:
- "coder": writing, reviewing or debugging code; algorithms; APIs.
- "writer": prose deliverables -- emails, docs, announcements, copy.
- "analyst": diagnosis, comparisons, trade-offs, "what should we do" questions.

Routes available: {list(SPECIALISTS)}.
Respond with ONLY this JSON: {{"route": "<route>", "reason": "<one short sentence>"}}
"""


def classify(client, goal: str) -> tuple[str, str]:
    """Return (route, reason); never fails -- see parse_route for the fallbacks."""
    reply = chat(
        client,
        [
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": goal},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    return parse_route(reply)


def parse_route(reply: str) -> tuple[str, str]:
    """JSON first; then a bare route name (small models often answer just
    "coder"); then the analyst fallback -- a router must always route."""
    data = extract_json(reply)
    if data is not None:
        route = str(data.get("route", "")).lower().strip()
        if route in SPECIALISTS:
            return route, str(data.get("reason", ""))
    mentioned = [r for r in SPECIALISTS if re.search(rf"\b{r}\b", reply, re.IGNORECASE)]
    if len(mentioned) == 1:
        return mentioned[0], "route name found in a non-JSON classifier reply"
    return "analyst", "fallback: classifier output was unparseable"


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[router] request: {goal}\n" + "-" * 72)

    route, reason = classify(client, goal)
    spec = SPECIALISTS[route]
    print(f"[route] -> {route} (temp {spec['temperature']}) -- {reason}")

    answer = chat(
        client,
        [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": goal},
        ],
        temperature=spec["temperature"],
        max_tokens=1400,
    )
    print("-" * 72 + f"\n[router] answered by the {route} specialist.")
    return answer


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nANSWER:\n{final}")
