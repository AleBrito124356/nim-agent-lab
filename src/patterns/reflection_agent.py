"""Reflection: generate -> self-critique against a rubric -> revise.

A writer model drafts, a critic model scores the draft against an explicit
rubric and lists concrete issues, and the writer revises using that critique.
The loop keeps the BEST-scoring version seen (not the last one -- revisions
sometimes regress), and stops early once the critic scores 9+.

The rubric is the load-bearing part. "Critique this" produces vague praise;
scoring named dimensions produces actionable edits.

Run standalone:
    python -m src.patterns.reflection_agent "Write a launch announcement for a CLI tool that diffs two Postgres schemas"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

DESCRIPTION = "Generate, self-critique against a rubric, revise; keeps the best-scoring draft."
DEFAULT_GOAL = (
    "Write a 150-word README introduction for an open-source library that "
    "retries failed HTTP requests with exponential backoff and jitter."
)

DEFAULT_ROUNDS = 2
TARGET_SCORE = 9

RUBRIC = """\
Score 1-10 overall, judging these dimensions:
1. Accuracy -- no invented facts, numbers or capabilities.
2. Clarity -- a reader gets the point in one pass; no filler.
3. Structure -- logical order, appropriate length for the request.
4. Tone -- professional and direct; no hype words.
5. Completeness -- everything the request asked for is present.
"""

WRITER_PROMPT = (
    "You are a senior technical writer. Produce exactly what the request "
    "asks for -- no preamble, no meta-commentary, just the deliverable."
)

CRITIC_PROMPT = f"""\
You are a demanding but fair editor. Critique the draft against this rubric:

{RUBRIC}
Respond with ONLY this JSON, no prose outside it:
{{"score": <1-10 integer>, "strengths": ["..."], "issues": ["..."], "suggestions": ["concrete edit 1", "..."]}}
"""

REVISER_PROMPT = (
    "You are the same technical writer, revising your draft. Apply every "
    "suggestion from the critique that improves the text; ignore any that "
    "would break the original request. Output ONLY the revised deliverable."
)


def _parse_critique(reply: str) -> dict:
    """Parse the critic's JSON; degrade gracefully to a neutral critique."""
    cleaned = re.sub(r"```(?:json)?", "", reply).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score", 5))
            return {
                "score": max(1, min(10, score)),
                "strengths": [str(s) for s in data.get("strengths", [])],
                "issues": [str(s) for s in data.get("issues", [])],
                "suggestions": [str(s) for s in data.get("suggestions", [])],
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"score": 5, "strengths": [], "issues": ["critique unparseable"], "suggestions": []}


def _critique(client, goal: str, draft: str) -> dict:
    reply = chat(
        client,
        [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"Request: {goal}\n\nDraft:\n{draft}"},
        ],
        temperature=0.0,
    )
    return _parse_critique(reply)


def run(goal: str, rounds: int = DEFAULT_ROUNDS) -> str:
    client = get_client()
    print(f"\n[reflection] goal: {goal}\n" + "-" * 72)

    draft = chat(
        client,
        [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": goal},
        ],
        temperature=0.7,
    )
    critique = _critique(client, goal, draft)
    best_draft, best_score = draft, critique["score"]
    print(f"[draft 0] score {best_score}/10")

    for round_no in range(1, rounds + 1):
        if critique["score"] >= TARGET_SCORE:
            print(f"[stop] score {critique['score']} >= {TARGET_SCORE}, no more rounds needed.")
            break

        issues = "\n".join(f"- {i}" for i in critique["issues"]) or "- (none listed)"
        suggestions = "\n".join(f"- {s}" for s in critique["suggestions"]) or "- (none listed)"
        print(f"[round {round_no}] issues:\n{issues}")

        draft = chat(
            client,
            [
                {"role": "system", "content": REVISER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original request: {goal}\n\nCurrent draft:\n{draft}\n\n"
                        f"Critique -- issues:\n{issues}\n\nSuggestions:\n{suggestions}"
                    ),
                },
            ],
            temperature=0.5,
        )
        critique = _critique(client, goal, draft)
        print(f"[round {round_no}] revised score {critique['score']}/10")

        if critique["score"] > best_score:
            best_draft, best_score = draft, critique["score"]

    print("-" * 72 + f"\n[reflection] best score: {best_score}/10")
    return best_draft


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nBEST DRAFT:\n{final}")
