"""Memory agent: short-term window + rolling summary + long-term fact store.

Three memory tiers, because each one fails alone:

- SHORT-TERM: the last few exchanges verbatim (recency, exact wording).
- ROLLING SUMMARY: when the window overflows, the oldest exchanges are
  compressed into a running summary instead of being dropped.
- LONG-TERM FACTS: after every turn an extractor pulls durable facts about
  the user ("prefers Postgres", "deadline is March 3") into a JSON file on
  disk. Facts are retrieved per-turn by word overlap between the new input
  and each fact's keywords (multi-word keywords such as "property
  management" are split into words), so only relevant memories enter the
  prompt. Facts already in the store are not stored twice.

The store survives restarts -- run the demo twice and the agent remembers.

When stdin is not a terminal, the default demo plays two scripted follow-up
turns so retrieval is visible in a single run. On a terminal (live backend)
it becomes an interactive chat instead.

Run standalone (interactive when attached to a terminal):
    python -m src.patterns.memory_agent "Hi, I'm Alejandro. I'm building a FastAPI backend and I prefer Postgres."
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, extract_json, get_client, is_offline  # noqa: E402

DESCRIPTION = "Short-term window, rolling summary compression, and a persistent JSON fact store."
DEFAULT_GOAL = (
    "Hi! I'm Alejandro, a full-stack developer in Panama City. I'm building "
    "a property-management app with FastAPI and I prefer Postgres over MySQL."
)
DEMO_FOLLOWUPS = [
    "Which database should I use for the tenant payments ledger?",
    "Any FastAPI tips for my property management project? It will run on a single small VPS.",
]

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
        w for w in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _normalize_fact(text: str) -> str:
    """Comparison key for de-duplication: case, spacing and punctuation-blind."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _clean_keywords(raw, fact: str) -> list[str]:
    """Keywords as a list of lowercase strings; derived from the fact if absent."""
    if isinstance(raw, str):
        raw = re.split(r"[,;]", raw)
    if not isinstance(raw, (list, tuple)):
        raw = []
    keywords = [str(k).strip().lower() for k in raw if str(k).strip()]
    return keywords or sorted(_tokens(fact))


# --------------------------------------------------------------------------
# Long-term store
# --------------------------------------------------------------------------


def load_facts() -> list[dict]:
    """Load the store, skipping malformed entries instead of crashing on them."""
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    if not isinstance(data, list):
        return []
    facts = []
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("fact"), str) and entry["fact"].strip():
            fact = entry["fact"].strip()
            facts.append({"fact": fact, "keywords": _clean_keywords(entry.get("keywords"), fact)})
    return facts


def save_facts(facts: list[dict]) -> None:
    """Write atomically: a crash mid-write must not corrupt the whole store."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_FILE.with_name(MEMORY_FILE.name + ".tmp")
    tmp.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, MEMORY_FILE)


def _keyword_tokens(entry: dict) -> set[str]:
    tokens: set[str] = set()
    for keyword in entry.get("keywords") or []:
        tokens |= _tokens(keyword)
    return tokens or _tokens(entry.get("fact", ""))


def retrieve_facts(facts: list[dict], query: str, k: int = TOP_K_FACTS) -> list[str]:
    """Rank stored facts by word overlap between the query and their keywords."""
    query_tokens = _tokens(query)
    scored = []
    for entry in facts:
        overlap = len(query_tokens & _keyword_tokens(entry))
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


def parse_facts(reply: str, known: list[dict]) -> list[dict]:
    """Turn the extractor reply into new, de-duplicated fact entries.

    Accepts the requested shape plus the variants models actually produce:
    bare strings instead of objects, "text" instead of "fact", keywords as a
    comma-separated string.
    """
    data = extract_json(reply)
    items = data.get("facts") if data is not None else None
    if not isinstance(items, list):
        return []
    seen = {_normalize_fact(f["fact"]) for f in known}
    results = []
    for item in items:
        if isinstance(item, str):
            fact, raw_keywords = item.strip(), None
        elif isinstance(item, dict):
            fact = str(item.get("fact") or item.get("text") or item.get("content") or "").strip()
            raw_keywords = item.get("keywords")
        else:
            continue
        key = _normalize_fact(fact)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append({"fact": fact, "keywords": _clean_keywords(raw_keywords, fact)})
    return results


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
    return parse_facts(reply, known)


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
        print(f"  [memory] compressed {len(overflow)} old message(s) into the rolling summary")

    def turn(self, user_msg: str) -> str:
        relevant = retrieve_facts(self.facts, user_msg)
        if relevant:
            print(f"  [memory] retrieved {len(relevant)} fact(s) for this turn:")
            for fact in relevant:
                print(f"    - {fact}")
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
        return reply


def run(goal: str) -> str:
    agent = MemoryAgent()
    print(f"\n[memory] store: {MEMORY_FILE} ({len(agent.facts)} fact(s) loaded)")
    print("-" * 72)

    print(f"\nyou> {goal}")
    reply = agent.turn(goal)
    print(f"agent> {reply}")

    if sys.stdin.isatty() and not is_offline():
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
    elif goal.strip() == DEFAULT_GOAL.strip():
        # Scripted follow-ups so one non-interactive run shows retrieval.
        for follow_up in DEMO_FOLLOWUPS:
            print(f"\nyou> {follow_up}")
            reply = agent.turn(follow_up)
            print(f"agent> {reply}")

    print("-" * 72 + f"\n[memory] session ended. {len(agent.facts)} fact(s) on disk.")
    return reply


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    run(goal)
