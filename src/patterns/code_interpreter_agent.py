"""Code interpreter: write Python, run it, read the error, fix it, repeat.

The agent writes a complete script that includes its own assert-based tests
and prints ALL TESTS PASSED on success. The script runs in a subprocess and
stdout/stderr are fed back (tail-truncated) until the tests pass or attempts
run out.

How the subprocess is isolated -- and how it is not:

- ``-I`` isolated mode: PYTHON* environment variables and the user
  site-packages directory are ignored, and the script's directory is not
  put on sys.path.
- ``-S``: the site module is not imported, so third-party packages in this
  venv (openai, pydantic...) are not importable. Standard library only.
- ``-X utf8``: UTF-8 stdio regardless of the console code page.
- A minimal, allowlisted environment (PATH, SYSTEMROOT, TEMP...). API keys,
  tokens and every other variable of the parent process -- including the
  NVIDIA_API_KEY that load_dotenv() put there -- are NOT passed on.
- stdin is /dev/null (input() fails fast instead of hanging), there is a
  hard timeout, and the temporary working directory is deleted afterwards.

SANDBOXING DISCLAIMER
---------------------
All of that is failure isolation, NOT a security boundary. The generated
code still runs with your OS user's permissions: it can read and write your
files and open network sockets. Fine for local experiments with a model you
chose; for anything multi-tenant or exposed to untrusted prompts, run the
execution step inside a container, a microVM (Firecracker/gVisor) or a
dedicated sandbox service.

Run standalone:
    python -m src.patterns.code_interpreter_agent "Write a function that returns the n-th Fibonacci number iteratively"
"""

from __future__ import annotations

import os
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
MAX_FEEDBACK_CHARS = 3000  # a runaway print loop must not flood the prompt
SUCCESS_MARKER = "ALL TESTS PASSED"

# The only variables generated code gets to see: enough to start Python on
# Windows/macOS/Linux, nothing secret.
ENV_ALLOWLIST = {
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
}

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

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str | None:
    match = _CODE_BLOCK_RE.search(reply)
    if match:
        return match.group(1).strip()
    # Some models skip the fence; accept raw output that looks like code.
    if "def " in reply or "assert " in reply:
        return reply.strip()
    return None


def child_env() -> dict[str, str]:
    """Allowlisted environment for the subprocess (no API keys, no tokens)."""
    return {key: value for key, value in os.environ.items() if key.upper() in ENV_ALLOWLIST}


def execute(code: str, workdir: Path) -> tuple[int, str, str]:
    """Run the script in an isolated subprocess. Returns (rc, stdout, stderr)."""
    script = workdir / "attempt.py"
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-X", "utf8", str(script)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            cwd=str(workdir),
            env=child_env(),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {TIMEOUT_SECONDS}s (infinite loop?)"


def _tail(text: str, limit: int = MAX_FEEDBACK_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} earlier characters truncated ...]\n" + text[-limit:]


def run(goal: str) -> str:
    client = get_client()
    print(f"\n[code-interpreter] task: {goal}")
    print("[code-interpreter] note: subprocess isolation is not a security sandbox.\n" + "-" * 72)

    messages: list[dict] = [
        {"role": "system", "content": CODER_PROMPT},
        {"role": "user", "content": f"Task: {goal}"},
    ]

    last_code = ""
    with tempfile.TemporaryDirectory(prefix="nim_code_interp_", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
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
                f"Exit code: {rc}\n--- stdout ---\n{_tail(stdout) or '(empty)'}\n"
                f"--- stderr ---\n{_tail(stderr) or '(empty)'}"
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
