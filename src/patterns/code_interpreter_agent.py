"""Code interpreter: write Python, run it, read the error, fix it, repeat.

The agent writes a complete script that includes its own assert-based tests
and prints ALL TESTS PASSED on success. The script runs in a subprocess with
a hard timeout and isolated mode (-I: no site-packages, no user config, no
inherited environment surprises). stdout/stderr are fed back verbatim and
the agent iterates until the tests pass or attempts run out.

SANDBOXING DISCLAIMER
---------------------
A subprocess with -I and a timeout is failure isolation, NOT a security
boundary. The generated code runs with your OS user's permissions: it can
read and write your files and open network sockets. Fine for local
experiments with a model you chose; for anything multi-tenant or exposed to
untrusted prompts, run the execution step inside a container, a microVM
(Firecracker/gVisor) or a dedicated sandbox service.

Run standalone:
    python -m src.patterns.code_interpreter_agent "Write a function that returns the n-th Fibonacci number iteratively"
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nim import chat, get_client  # noqa: E402

DESCRIPTION = "Writes Python with its own tests, executes in a subprocess, iterates until green."
DEFAULT_GOAL = (
    "Write a function slugify(text) that lowercases, converts spaces and "
    "underscores to hyphens, strips all other non-alphanumeric characters, "
    "and collapses repeated hyphens."
)

MAX_ATTEMPTS = 4
TIMEOUT_SECONDS = 10
SUCCESS_MARKER = "ALL TESTS PASSED"

CODER_PROMPT = f"""\
You are a careful Python programmer. Write ONE complete, standard-library-only
script that solves the task. Requirements:
- Include at least 5 assert-based tests covering normal and edge cases.
- End the script with: print("{SUCCESS_MARKER}")  (only reached if all asserts pass).
- No input(), no network access, no file writes.
Respond with ONLY one fenced code block:
```python
<the full script>
```
"""

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str | None:
    match = _CODE_BLOCK_RE.search(reply)
    if match:
        return match.group(1).strip()
    # Some models skip the fence; accept raw output that looks like code.
    if "def " in reply or "assert " in reply:
        return reply.strip()
    return None


def execute(code: str, workdir: Path) -> tuple[int, str, str]:
    """Run the script in an isolated subprocess. Returns (rc, stdout, stderr)."""
    script = workdir / "attempt.py"
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(workdir),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {TIMEOUT_SECONDS}s (infinite loop?)"


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[code-interpreter] task: {goal}")
    print(f"[code-interpreter] note: subprocess isolation is not a security sandbox.\n" + "-" * 72)

    workdir = Path(tempfile.mkdtemp(prefix="nim_code_interp_"))
    messages: list[dict] = [
        {"role": "system", "content": CODER_PROMPT},
        {"role": "user", "content": f"Task: {goal}"},
    ]

    last_code = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = chat(client, messages, temperature=0.2, max_tokens=1600)
        code = extract_code(reply)
        if code is None:
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": "No code block found. Respond with ONLY one ```python fenced block.",
                }
            )
            print(f"[attempt {attempt}] no code block in reply; asking again.")
            continue

        last_code = code
        print(f"\n[attempt {attempt}] running {len(code.splitlines())} lines...")
        rc, stdout, stderr = execute(code, workdir)

        if rc == 0 and SUCCESS_MARKER in stdout:
            print(f"[attempt {attempt}] PASS\n" + "-" * 72)
            print("[code-interpreter] final working script:\n")
            print(code)
            return code

        feedback = (
            f"Exit code: {rc}\n--- stdout ---\n{stdout.strip() or '(empty)'}\n"
            f"--- stderr ---\n{stderr.strip() or '(empty)'}"
        )
        print(f"[attempt {attempt}] FAIL\n{feedback}")
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The script failed.\n{feedback}\n\n"
                    "Fix the bug and resend the COMPLETE corrected script in one "
                    "```python block. Keep the tests."
                ),
            }
        )

    print("-" * 72 + "\n[code-interpreter] gave up; returning last attempt.")
    return last_code or "The agent produced no runnable code."


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL
    run(goal)
