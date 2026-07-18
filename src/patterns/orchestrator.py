"""Orchestrator: a supervisor delegates sub-tasks to other patterns as workers.

The supervisor decomposes the goal into at most four sub-tasks and assigns
each to the best-suited worker, then merges the results:

- "react"      -> the ReAct agent (src/patterns/react_agent.py): anything
                  needing tools -- unit conversions, arithmetic, lookups.
- "reflection" -> the reflection agent (src/patterns/reflection_agent.py):
                  prose deliverables that benefit from a critique pass.
- "direct"     -> a plain single LLM call: simple factual or list sub-tasks
                  where extra machinery is just latency.

This is the pattern the other eleven feed into: real systems are rarely one
agent, they are a supervisor and a bench of cheap specialists.

Run standalone:
    python -m src.patterns.orchestrator "Prepare a briefing on switching our fleet reports from imperial to metric"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402
from src.patterns import react_agent, reflection_agent  # noqa: E402

DESCRIPTION = "Supervisor decomposes the goal, delegates to other patterns as workers, merges results."
DEFAULT_GOAL = (
    "Prepare a short internal briefing: convert our delivery van's 18 mpg "
    "fuel economy figure of 120 miles per day into kilometers, estimate "
    "monthly distance for 22 working days, and write a 3-bullet summary "
    "of why the ops team should track kilometers going forward."
)

MAX_SUBTASKS = 4

SUPERVISOR_PROMPT = f"""\
You are a supervisor coordinating specialist workers. Decompose the goal
into 2 to {MAX_SUBTASKS} INDEPENDENT sub-tasks and assign each to a worker:

- "react": calculations, unit conversions, anything needing tools.
- "reflection": polished prose deliverables (summaries, briefings, emails).
- "direct": simple factual or listing sub-tasks.

Each sub-task description must be fully self-contained -- workers cannot
see the original goal or each other's output.
Respond with ONLY this JSON:
{{"subtasks": [{{"worker": "<react|reflection|direct>", "task": "<self-contained instruction>"}}]}}
"""


def _direct_worker(task: str) -> str:
    client = get_client()
    return chat(
        client,
        [
            {"role": "system", "content": "Answer the task directly and concisely."},
            {"role": "user", "content": task},
        ],
        temperature=0.3,
    )


def _reflection_worker(task: str) -> str:
    # One critique round keeps cost sane inside a multi-agent run.
    return reflection_agent.run(task, rounds=1)


WORKERS = {
    "react": react_agent.run,
    "reflection": _reflection_worker,
    "direct": _direct_worker,
}


def _parse_subtasks(reply: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", reply).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    subtasks = []
    for item in data.get("subtasks", [])[:MAX_SUBTASKS]:
        worker = str(item.get("worker", "")).lower().strip()
        task = str(item.get("task", "")).strip()
        if task:
            subtasks.append({"worker": worker if worker in WORKERS else "direct", "task": task})
    return subtasks


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[orchestrator] goal: {goal}\n" + "=" * 72)

    reply = chat(
        client,
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": goal},
        ],
        temperature=0.0,
    )
    subtasks = _parse_subtasks(reply)
    if not subtasks:
        subtasks = [{"worker": "direct", "task": goal}]

    print("[delegation plan]")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. [{st['worker']}] {st['task']}")

    results: list[tuple[str, str, str]] = []
    for i, st in enumerate(subtasks, 1):
        print(f"\n{'=' * 72}\n[worker {i}/{len(subtasks)}: {st['worker']}] starting...")
        try:
            output = WORKERS[st["worker"]](st["task"])
        except SystemExit:
            raise  # missing API key: propagate the friendly exit
        except Exception as exc:
            output = f"Worker failed: {exc}"
        results.append((st["worker"], st["task"], output))
        print(f"[worker {i}] done.")

    merged_input = "\n\n".join(
        f"Sub-task ({worker}): {task}\nResult:\n{output}"
        for worker, task, output in results
    )
    final = chat(
        client,
        [
            {
                "role": "system",
                "content": (
                    "You are the supervisor. Merge the worker results into ONE "
                    "coherent final answer to the original goal. Resolve overlaps, "
                    "keep every number the workers computed, add nothing new."
                ),
            },
            {"role": "user", "content": f"Original goal: {goal}\n\n{merged_input}"},
        ],
        temperature=0.2,
        max_tokens=1400,
    )
    print("\n" + "=" * 72 + "\n[orchestrator] merged final answer ready.")
    return final


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    final = run(goal)
    print(f"\nFINAL ANSWER:\n{final}")
