"""Memory agent: short-term window + rolling summary + long-term fact store.

Three memory tiers, because each one fails alone:

- SHORT-TERM: the last few exchanges verbatim (recency, exact wording).
- ROLLING SUMMARY: when the window overflows, the oldest exchanges are
  compressed into a running summary instead of being dropped.
- LONG-TERM FACTS: after every turn an extractor pulls durable facts about
  the user ("prefers Postgres", "deadline is March 3") into a JSON file on
  disk. Facts are retrieved per-turn by keyword overlap with the new input,
  so only relevant memories enter the prompt.

The store survives restarts -- run the demo twice and the agent remembers.

Run standalone (interactive when attached to a terminal):
    python -m src.patterns.memory_agent "Hi, I'm Alejandro. I'm building a FastAPI backend and I prefer Postgres."
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

DESCRIPTION = "Short-term window, rolling summary compression, and a persistent JSON fact store."
DEFAULT_GOAL = (
    "Hi! I'm Alejandro, a full-stack developer in Panama City. I'm building "
    "a property-management app with FastAPI and I prefer Postgres over MySQL."
)

MEMORY_FILE = Path(__file__).resolve().parents[2] / "memory_store.json"
SHORT_TERM_MAX_MESSAGES = 8   # user+assistant messages kept verbatim
TOP_K_FACTS = 5

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "i", "im", "is", "are", "was", "to",
    "of", "in", "on", "for", "with", "my", "me", "you", "it", "that", "this",
    "at", "be", "do", "have", "what", "how", "your", "am", "we", "they",
}


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


# --------------------------------------------------------------------------
# Long-term store
# --------------------------------------------------------------------------


def load_facts() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_facts(facts: list[dict]) -> None:
    MEMORY_FILE.write_text(
        json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def retrieve_facts(facts: list[dict], query: str, k: int = TOP_K_FACTS) -> list[str]:
    """Rank stored facts by keyword overlap with the incoming message."""
    query_tokens = _tokens(query)
    scored = []
    for entry in facts:
        overlap = len(query_tokens & set(entry.get("keywords", [])))
        if overlap > 0:
            scored.append((overlap, entry["fact"]))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:k]]


EXTRACTOR_PROMPT = """\
Extract durable facts about the user from this exchange -- preferences,
projects, names, deadlines, constraints. Skip anything transient or already
in the known-facts list. Respond with ONLY this JSON:
{"facts": [{"fact": "<one sentence>", "keywords": ["lowercase", "search", "terms"]}]}
Respond {"facts": []} if there is nothing new worth keeping.
"""


def extract_facts(client, user_msg: str, assistant_msg: str, known: list[dict]) -> list[dict]:
    known_lines = "\n".join(f"- {f['fact']}" for f in known[-30:]) or "(none)"
    reply = chat(
        client,
        [
            {"role": "system", "content": EXTRACTOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Known facts:\n{known_lines}\n\n"
                    f"User said: {user_msg}\nAssistant replied: {assistant_msg}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=400,
    )
    cleaned = re.sub(r"```(?:json)?", "", reply).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    results = []
    for item in data.get("facts", []):
        fact = str(item.get("fact", "")).strip()
        if not fact:
            continue
        keywords = [str(k).lower() for k in item.get("keywords", [])] or sorted(_tokens(fact))
        results.append({"fact": fact, "keywords": keywords})
    return results


# --------------------------------------------------------------------------
# Conversation loop
# --------------------------------------------------------------------------

AGENT_PROMPT = """\
You are a helpful personal assistant with memory. Use the rolling summary
and the retrieved long-term facts naturally -- reference what you know about
the user without listing it back robotically. Be concise.
"""


class MemoryAgent:
    def __init__(self) -> None:
        self.client = get_client()
        self.window: list[dict] = []      # verbatim recent messages
        self.summary: str = ""            # rolling compressed history
        self.facts: list[dict] = load_facts()

    def _compress_overflow(self) -> None:
        """Fold the oldest exchanges into the rolling summary."""
        if len(self.window) <= SHORT_TERM_MAX_MESSAGES:
            return
        overflow = self.window[:-SHORT_TERM_MAX_MESSAGES]
        self.window = self.window[-SHORT_TERM_MAX_MESSAGES:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in overflow)
        self.summary = chat(
            self.client,
            [
                {
                    "role": "system",
                    "content": (
                        "Update the running conversation summary. Keep it under "
                        "120 words, factual, third person."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current summary:\n{self.summary or '(empty)'}\n\n"
                        f"New exchanges to fold in:\n{transcript}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=250,
        )

    def turn(self, user_msg: str) -> str:
        relevant = retrieve_facts(self.facts, user_msg)
        context_parts = []
        if self.summary:
            context_parts.append(f"Conversation summary so far:\n{self.summary}")
        if relevant:
            context_parts.append(
                "Relevant long-term facts about the user:\n"
                + "\n".join(f"- {f}" for f in relevant)
            )
        system = AGENT_PROMPT + ("\n\n" + "\n\n".join(context_parts) if context_parts else "")

        messages = [{"role": "system", "content": system}, *self.window,
                    {"role": "user", "content": user_msg}]
        reply = chat(self.client, messages, temperature=0.4)

        self.window.append({"role": "user", "content": user_msg})
        self.window.append({"role": "assistant", "content": reply})
        self._compress_overflow()

        new_facts = extract_facts(self.client, user_msg, reply, self.facts)
        if new_facts:
            self.facts.extend(new_facts)
            save_facts(self.facts)
            for f in new_facts:
                print(f"  [memory] stored: {f['fact']}")
        if relevant:
            print(f"  [memory] retrieved {len(relevant)} fact(s) for this turn")
        return reply


def run(goal: str) -> str:
    agent = MemoryAgent()
    print(f"\n[memory] store: {MEMORY_FILE} ({len(agent.facts)} fact(s) loaded)")
    print("-" * 72)

    print(f"\nyou> {goal}")
    reply = agent.turn(goal)
    print(f"agent> {reply}")

    if sys.stdin.isatty():
        print("\n(interactive mode -- type 'exit' to quit)")
        while True:
            try:
                user_msg = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_msg or user_msg.lower() in {"exit", "quit"}:
                break
            reply = agent.turn(user_msg)
            print(f"agent> {reply}")

    print("-" * 72 + f"\n[memory] session ended. {len(agent.facts)} fact(s) on disk.")
    return reply


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    run(goal)
