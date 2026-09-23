#!/usr/bin/env python3
"""nim-agent-lab CLI: run any of the 12 agent patterns from one entry point.

    python main.py --list
    python main.py react
    python main.py react --offline                      # no key, no network
    python main.py debate --offline --trace traces/debate.jsonl
    python main.py planner-executor --goal "Budget a 3-day offsite for 12 people"
    python main.py router --record cassettes/router.json # live run -> new cassette
    python main.py --all --offline                      # every pattern, side by side
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
import tempfile
from pathlib import Path

from src import nim

# name -> (module path, static description). Descriptions live here so
# --list works instantly without importing every pattern module.
PATTERNS: dict[str, tuple[str, str]] = {
    "react": (
        "src.patterns.react_agent",
        "Thought -> Action -> Observation loop with calculator, unit converter and weather tools.",
    ),
    "tool-calling": (
        "src.patterns.tool_calling_agent",
        "Native OpenAI-style function calling with JSON-schema tools and parallel call handling.",
    ),
    "planner-executor": (
        "src.patterns.planner_executor",
        "Planner decomposes the goal, executor runs each step, replanner revises after every result.",
    ),
    "reflection": (
        "src.patterns.reflection_agent",
        "Generate, self-critique against a rubric, revise; keeps the best-scoring draft.",
    ),
    "debate": (
        "src.patterns.debate_agents",
        "Two agents argue opposite sides across two rounds; a judge synthesizes the answer.",
    ),
    "router": (
        "src.patterns.router_agent",
        "Classifies each request and routes it to a coder, writer or analyst persona.",
    ),
    "memory": (
        "src.patterns.memory_agent",
        "Short-term window, rolling summary compression, and a persistent JSON fact store.",
    ),
    "guardrails": (
        "src.patterns.guardrails_agent",
        "Injection heuristics + LLM moderation on input; PII scrub + schema enforcement on output.",
    ),
    "code-interpreter": (
        "src.patterns.code_interpreter_agent",
        "Writes Python with its own tests, executes in a subprocess, iterates until green.",
    ),
    "structured-output": (
        "src.patterns.structured_output_agent",
        "Pydantic schema -> JSON extraction with a validation-error repair loop.",
    ),
    "human-in-the-loop": (
        "src.patterns.human_in_the_loop",
        "Agent proposes actions; each needs terminal approval; every decision is audit-logged.",
    ),
    "orchestrator": (
        "src.patterns.orchestrator",
        "Supervisor delegates to ReAct, reflection and debate workers, then merges the results.",
    ),
}


def cassette_for(pattern: str) -> Path:
    return nim.CASSETTE_DIR / f"{pattern}.json"


def list_patterns() -> None:
    width = max(len(name) for name in PATTERNS)
    print("\nAvailable patterns:\n")
    for name, (_, description) in PATTERNS.items():
        print(f"  {name:<{width}}  {description}")
    print(
        "\nRun one:        python main.py <pattern> [--goal \"...\"]\n"
        "No API key?     python main.py <pattern> --offline\n"
        "Compare all:    python main.py --all --offline\n"
        "Each pattern also runs standalone: python -m src.patterns.react_agent\n"
    )


def _replay_status() -> str | None:
    from src.replay import ReplayClient

    replay = nim.find_layer(ReplayClient)
    if replay is None:
        return None
    used, total = replay.position, replay.total
    note = "" if used == total else " (this run took a shorter path than the cassette)"
    return f"[offline] replayed {used}/{total} cassette replies from {_rel(Path(replay.source))}{note}"


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def run_pattern(args: argparse.Namespace) -> int:
    module_path, _ = PATTERNS[args.pattern]
    module = importlib.import_module(module_path)
    default_goal = getattr(module, "DEFAULT_GOAL", "")
    goal = args.goal or default_goal

    if args.offline:
        nim.configure(backend="mock", cassette=cassette_for(args.pattern), trace_path=args.trace)
        print(f"[offline] replaying {_rel(cassette_for(args.pattern))} -- no API key, no network.")
        if args.goal and args.goal.strip() != default_goal.strip():
            print(
                "[offline] note: a cassette replays the replies recorded for the pattern's "
                "demo goal,\n          so the model's answers below follow that goal, not "
                "yours. Tools, parsing\n          and control flow still run for real. Drop "
                "--offline to ask a live model."
            )
    elif args.record:
        nim.configure(
            backend="record",
            record_path=args.record,
            trace_path=args.trace,
            meta={"pattern": args.pattern, "goal": goal},
        )
    else:
        nim.configure(trace_path=args.trace)  # live (or whatever NIM_BACKEND says)

    from src.replay import RecordingClient, ReplayError
    from src.trace import TracingClient

    try:
        result = module.run(goal)
    except ReplayError as exc:
        sys.stderr.write(f"\n[offline] cassette problem: {exc}\n")
        return 2

    if result:
        print(f"\n{'=' * 72}\nRESULT ({args.pattern}):\n{result}")

    status = _replay_status()
    if status:
        print(f"\n{status}")
    recorder = nim.find_layer(RecordingClient)
    if recorder is not None:
        count = len(recorder.cassette["interactions"])
        print(f"\n[record] saved {count} interaction(s) to {recorder.path}")
        print(f"         replay it with: NIM_BACKEND=mock NIM_CASSETTE={recorder.path} "
              f"python -m {module_path}")
    tracer = nim.find_layer(TracingClient)
    if tracer is not None:
        print("\n" + tracer.format_summary(title=f"LLM trace ({args.pattern})"))
    return 0


def run_all(show_output: bool) -> int:
    """Replay every pattern's cassette and print one comparison table."""
    from src.replay import ReplayClient
    from src.trace import TracingClient

    rows = []
    for name, (module_path, _) in PATTERNS.items():
        module = importlib.import_module(module_path)
        nim.configure(backend="mock", cassette=cassette_for(name), trace_path=nim.MEMORY_TRACE)
        captured = io.StringIO()
        status = "ok"
        # Dry-run semantics for patterns that would otherwise prompt a human,
        # and a scratch dir for runtime files: the table is reproducible and
        # your real memory store / audit log stay untouched.
        with contextlib.redirect_stdout(captured), _stdin_closed(), _scratch_runtime_files():
            try:
                module.run(module.DEFAULT_GOAL)
            except Exception as exc:  # report the failure in the table, keep going
                status = f"FAILED: {type(exc).__name__}: {exc}"
        if show_output:
            print(captured.getvalue())
        tracer = nim.find_layer(TracingClient)
        replay = nim.find_layer(ReplayClient)
        totals = tracer.summary()["totals"] if tracer else {"calls": 0, "prompt_tokens": 0,
                                                             "completion_tokens": 0}
        if status == "ok" and replay is not None and replay.remaining:
            status = f"ok ({replay.remaining} cassette replies unused)"
        rows.append((name, totals["calls"], totals["prompt_tokens"], totals["completion_tokens"], status))
    nim.reset()

    width = max(len(row[0]) for row in rows)
    print(f"\n{'pattern':<{width}}  calls  prompt tok  compl. tok  status")
    print(f"{'-' * width}  -----  ----------  ----------  ------")
    for name, calls, prompt, completion, status in rows:
        print(f"{name:<{width}}  {calls:>5}  {prompt:>10,}  {completion:>10,}  {status}")
    print(f"{'total':<{width}}  {sum(r[1] for r in rows):>5}  {sum(r[2] for r in rows):>10,}  "
          f"{sum(r[3] for r in rows):>10,}")
    print(
        "\nOffline replay of cassettes/*.json. Token counts are estimated (~4 chars/token)\n"
        "from the real prompts each pattern built, because hand-authored cassettes carry\n"
        "no provider usage; a cassette recorded with --record reports real usage."
    )
    return 0 if all(row[4].startswith("ok") for row in rows) else 1


@contextlib.contextmanager
def _stdin_closed():
    original = sys.stdin
    sys.stdin = io.StringIO("")
    try:
        yield
    finally:
        sys.stdin = original


@contextlib.contextmanager
def _scratch_runtime_files():
    """Point the memory store, audit log and HITL workspace at a temp dir."""
    from src.patterns import human_in_the_loop, memory_agent

    targets = [
        (memory_agent, "MEMORY_FILE", "memory_store.json"),
        (human_in_the_loop, "AUDIT_LOG", "audit_log.jsonl"),
        (human_in_the_loop, "WORKSPACE", "hitl_workspace"),
    ]
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in targets]
    with tempfile.TemporaryDirectory(prefix="nim_all_", ignore_cleanup_errors=True) as tmp:
        for module, attr, name in targets:
            setattr(module, attr, Path(tmp) / name)
        try:
            yield
        finally:
            for module, attr, value in originals:
                setattr(module, attr, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nim-agent-lab",
        description="12 production-grade AI agent patterns on free NVIDIA NIM APIs.",
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        choices=sorted(PATTERNS),
        help="pattern to run (see --list)",
    )
    parser.add_argument("--goal", help="task for the agent (defaults to the pattern's demo goal)")
    parser.add_argument("--list", action="store_true", help="describe all patterns and exit")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--offline",
        action="store_true",
        help="replay cassettes/<pattern>.json instead of calling NIM: no API key, no network",
    )
    backend.add_argument(
        "--record",
        metavar="PATH",
        help="call NIM for real and save every exchange to a cassette at PATH",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE.jsonl",
        help="log every LLM call (messages, params, latency, tokens) and print a summary",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every pattern and print a calls/tokens comparison table (needs --offline)",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="with --all: also print each pattern's full output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    nim.safe_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.all:
        if not args.offline:
            parser.error("--all only runs with --offline (live, it would make ~70 API calls)")
        if args.pattern or args.goal or args.trace:
            parser.error("--all runs every demo goal; it takes no pattern, --goal or --trace")
        return run_all(args.show_output)

    if args.list or not args.pattern:
        list_patterns()
        return 0

    return run_pattern(args)


if __name__ == "__main__":
    sys.exit(main())
