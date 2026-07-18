"""Debate: two agents argue opposite sides, a judge synthesizes the answer.

Self-consistency through structured disagreement. One model instance is
forced to argue FOR the position implied by the question, another AGAINST it,
each getting an opening statement and one rebuttal. A judge then writes the
final answer from the full transcript -- typically more balanced and better
caveated than a single-pass response, because the weak points of each side
were surfaced explicitly.

Run standalone:
    python -m src.patterns.debate_agents "Should a 5-person startup build its own auth instead of using a managed provider?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

DESCRIPTION = "Two agents argue opposite sides across two rounds; a judge synthesizes the final answer."
DEFAULT_GOAL = (
    "Should a small SaaS team migrate its monolith to microservices when it "
    "reaches 50k monthly active users?"
)

ADVOCATE_PROMPT = """\
You are the PRO advocate in a structured debate. Argue the strongest honest
case IN FAVOR of the position in the question. Use concrete reasoning and
real-world trade-offs. Never concede the debate; never argue the other side.
Keep each statement under 180 words.
"""

SKEPTIC_PROMPT = """\
You are the CON advocate in a structured debate. Argue the strongest honest
case AGAINST the position in the question. Use concrete reasoning and
real-world trade-offs. Never concede the debate; never argue the other side.
Keep each statement under 180 words.
"""

JUDGE_PROMPT = """\
You are an impartial judge. You are given a debate transcript with a PRO and
a CON side. Write the final answer for the user:
1. State the most defensible position (may be conditional: "it depends on X").
2. Give the 2-3 strongest points from EACH side that survived rebuttal.
3. End with one concrete recommendation.
Do not mention the debate mechanics or the word "transcript".
"""


def _statement(client, system_prompt: str, question: str, transcript: str, instruction: str) -> str:
    return chat(
        client,
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Debate question: {question}\n\n"
                    f"Transcript so far:\n{transcript or '(debate is just starting)'}\n\n"
                    f"{instruction}"
                ),
            },
        ],
        temperature=0.6,
    )


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[debate] question: {goal}\n" + "-" * 72)

    transcript = ""

    pro_opening = _statement(
        client, ADVOCATE_PROMPT, goal, transcript, "Give your opening statement."
    )
    transcript += f"PRO (opening):\n{pro_opening}\n\n"
    print(f"\nPRO (opening):\n{pro_opening}")

    con_opening = _statement(
        client, SKEPTIC_PROMPT, goal, transcript, "Give your opening statement."
    )
    transcript += f"CON (opening):\n{con_opening}\n\n"
    print(f"\nCON (opening):\n{con_opening}")

    pro_rebuttal = _statement(
        client,
        ADVOCATE_PROMPT,
        goal,
        transcript,
        "Rebut the CON side's strongest points directly.",
    )
    transcript += f"PRO (rebuttal):\n{pro_rebuttal}\n\n"
    print(f"\nPRO (rebuttal):\n{pro_rebuttal}")

    con_rebuttal = _statement(
        client,
        SKEPTIC_PROMPT,
        goal,
        transcript,
        "Rebut the PRO side's strongest points directly.",
    )
    transcript += f"CON (rebuttal):\n{con_rebuttal}\n\n"
    print(f"\nCON (rebuttal):\n{con_rebuttal}")

    verdict = chat(
        client,
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"Debate question: {goal}\n\nFull transcript:\n{transcript}",
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    print("-" * 72 + "\n[debate] judge has ruled.")
    return verdict


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nVERDICT:\n{final}")
