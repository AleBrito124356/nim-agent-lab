#!/usr/bin/env python3
"""nim-agent-lab CLI: run any of the 12 agent patterns from one entry point.

    python main.py --list
    python main.py react
    python main.py planner-executor --goal "Budget a 3-day offsite for 12 people"
"""

from __future__ import annotations

import argparse
import importlib
import sys

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
        "Supervisor decomposes the goal, delegates to other patterns as workers, merges results.",
    ),
}


def list_patterns() -> None:
    width = max(len(name) for name in PATTERNS)
    print("\nAvailable patterns:\n")
    for name, (_, description) in PATTERNS.items():
        print(f"  {name:<{width}}  {description}")
    print(
        "\nRun one:  python main.py <pattern> [--goal \"...\"]\n"
        "Each pattern also runs standalone: python -m src.patterns.react_agent\n"
    )


def main() -> int:
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
    args = parser.parse_args()

    if args.list or not args.pattern:
        list_patterns()
        return 0

    module_path, _ = PATTERNS[args.pattern]
    module = importlib.import_module(module_path)
    goal = args.goal or getattr(module, "DEFAULT_GOAL", "")

    result = module.run(goal)
    if result:
        print(f"\n{'=' * 72}\nRESULT ({args.pattern}):\n{result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
