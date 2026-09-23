"""Orchestrator: a supervisor delegates sub-tasks to other patterns as workers.

The supervisor decomposes the goal into at most four sub-tasks and assigns
each to the best-suited worker, then merges the results:

- "react"      -> the ReAct agent (src/patterns/react_agent.py): anything
                  needing tools -- unit conversions, arithmetic, lookups.
- "reflection" -> the reflection agent (src/patterns/reflection_agent.py):
                  prose deliverables that benefit from a critique pass.
- "debate"     -> the debate agents (src/patterns/debate_agents.py):
                  contested judgment calls where both sides need a hearing.
- "direct"     -> a plain single LLM call: simple factual or list sub-tasks
                  where extra machinery is just latency.

Workers share the process-wide client (src/nim.get_client), so a single
trace or cassette covers the supervisor and every nested worker call.

This is where the other patterns compose: real systems are rarely one agent,
they are a supervisor and a bench of cheap specialists.

Run standalone:
    python -m src.patterns.orchestrator "Prepare a briefing on switching our fleet reports from imperial to metric"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import BackendError, chat, extract_json, get_client  # noqa: E402
from src.patterns import debate_agents, react_agent, reflection_agent  # noqa: E402

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
- "debate": contested judgment calls where both sides deserve a hearing.
- "direct": simple factual or listing sub-tasks.

Each sub-task description must be fully self-contained -- workers cannot
see the original goal or each other's output.
Respond with ONLY this JSON:
{{"subtasks": [{{"worker": "<react|reflection|debate|direct>", "task": "<self-contained instruction>"}}]}}
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
    "debate": debate_agents.run,
    "direct": _direct_worker,
}

# Names models use for the workers when they paraphrase the prompt.
_WORKER_ALIASES = {
    "tools": "react", "tool": "react", "calculator": "react", "react_agent": "react",
    "writer": "reflection", "write": "reflection", "prose": "reflection",
    "debater": "debate", "llm": "direct", "plain": "direct", "answer": "direct",
}


def _parse_subtasks(reply: str) -> list[dict]:
    """Accept the requested shape plus bare-string sub-tasks and alias keys
    ("description" for "task", "agent" for "worker"). Unknown or missing
    workers fall back to "direct"."""
    data = extract_json(reply)
    items = data.get("subtasks") if data is not None else None
    if not isinstance(items, list):
        return []
    subtasks = []
    for item in items:
        if isinstance(item, str):
            worker, task = "direct", item.strip()
        elif isinstance(item, dict):
            worker = str(item.get("worker") or item.get("agent") or "").lower().strip()
            task = str(item.get("task") or item.get("description") or item.get("instruction") or "").strip()
        else:
            continue
        worker = _WORKER_ALIASES.get(worker, worker)
        if task:
            subtasks.append({"worker": worker if worker in WORKERS else "direct", "task": task})
        if len(subtasks) == MAX_SUBTASKS:
            break
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
        print("[orchestrator] supervisor reply had no usable sub-tasks; answering directly.")
        subtasks = [{"worker": "direct", "task": goal}]

    print("[delegation plan]")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. [{st['worker']}] {st['task']}")

    results: list[tuple[str, str, str]] = []
    for i, st in enumerate(subtasks, 1):
        print(f"\n{'=' * 72}\n[worker {i}/{len(subtasks)}: {st['worker']}] starting...")
        try:
            output = WORKERS[st["worker"]](st["task"])
        except (SystemExit, BackendError):
            raise  # missing key or a broken backend: never disguise as a worker result
        except Exception as exc:
            output = f"Worker failed: {type(exc).__name__}: {exc}"
            print(f"[worker {i}] {output}")
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
