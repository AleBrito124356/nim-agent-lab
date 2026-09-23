from __future__ import annotations

from pathlib import Path

import pytest

from src.patterns import code_interpreter_agent as ci

PASSING = 'assert 1 + 1 == 2\nprint("ALL TESTS PASSED")'


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("```python\nprint(1)\n```", "print(1)"),
        ("Here you go:\n```py\nx = 1\n```\nDone.", "x = 1"),
        ("```\nassert True\n```", "assert True"),
        ("def f():\n    return 1", "def f():\n    return 1"),
        ("Sorry, I cannot help with that.", None),
    ],
)
def test_extract_code(reply, expected):
    assert ci.extract_code(reply) == expected


def test_execute_pass(tmp_path):
    rc, out, err = ci.execute(PASSING, tmp_path)
    assert rc == 0 and "ALL TESTS PASSED" in out and err == ""


def test_execute_fail_reports_traceback(tmp_path):
    rc, _, err = ci.execute("assert 1 == 2, 'math is broken'", tmp_path)
    assert rc == 1 and "AssertionError: math is broken" in err


def test_execute_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "TIMEOUT_SECONDS", 1)
    rc, _, err = ci.execute("while True:\n    pass", tmp_path)
    assert rc == -1 and "Timed out" in err


def test_generated_code_cannot_see_secrets(tmp_path, monkeypatch):
    # audit: the child inherited NVIDIA_API_KEY from the parent environment
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-FAKE-PROBE-VALUE")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@host/db")
    code = (
        "import os\n"
        "print([os.environ.get(k) for k in ('NVIDIA_API_KEY', 'GITHUB_TOKEN', 'DATABASE_URL')])"
    )
    rc, out, _ = ci.execute(code, tmp_path)
    assert rc == 0 and out.strip() == "[None, None, None]"


def test_generated_code_cannot_import_site_packages(tmp_path):
    # audit: -I alone still exposed the venv's site-packages (pydantic imported fine)
    rc, _, err = ci.execute("import pydantic", tmp_path)
    assert rc == 1 and "ModuleNotFoundError" in err


def test_stdin_is_closed_and_output_is_utf8(tmp_path):
    rc, out, err = ci.execute('print("→ ok")\ninput()', tmp_path)
    assert "→ ok" in out
    assert rc == 1 and "EOFError" in err


def test_feedback_is_tail_truncated():
    long = "x" * 5000 + "THE END"
    tail = ci._tail(long, limit=100)
    assert tail.endswith("THE END") and "truncated" in tail and len(tail) < 200


def test_run_iterates_until_green_and_cleans_its_workdir(scripted, monkeypatch):
    seen: list[Path] = []
    real_execute = ci.execute

    def spy(code, workdir):
        seen.append(workdir)
        return real_execute(code, workdir)

    monkeypatch.setattr(ci, "execute", spy)
    client = scripted(
        "```python\nassert 'a'.upper() == 'b'\nprint('ALL TESTS PASSED')\n```",
        f"```python\n{PASSING}\n```",
    )
    assert ci.run("anything") == PASSING
    assert "AssertionError" in client.requests[1]["messages"][-1]["content"]
    assert seen and not seen[0].exists()  # audit: mkdtemp dir used to be left behind


def test_run_gives_up_after_max_attempts(scripted, capsys):
    scripted(*["```python\nraise SystemExit(3)\n```"] * ci.MAX_ATTEMPTS)
    assert ci.run("impossible") == "raise SystemExit(3)"
    assert "gave up" in capsys.readouterr().out


def test_reply_without_code_asks_again(scripted):
    client = scripted("I'd rather explain it in words.", f"```python\n{PASSING}\n```")
    ci.run("anything")
    assert "No code block found" in client.requests[1]["messages"][-1]["content"]
