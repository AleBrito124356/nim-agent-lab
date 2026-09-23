from __future__ import annotations

import json

import pytest

import main
from src import nim, trace
from src.replay import ReplayClient
from src.trace import TracingClient
from tests.helpers import CASSETTES, read_jsonl


def totals(records):
    return (sum(r["usage"]["prompt_tokens"] for r in records),
            sum(r["usage"]["completion_tokens"] for r in records))


def test_debate_trace_has_one_line_per_call_and_a_matching_summary(tmp_path, capsys):
    path = tmp_path / "traces" / "debate.jsonl"
    assert main.main(["debate", "--offline", "--trace", str(path)]) == 0
    records = read_jsonl(path)
    assert len(records) == 5  # 4 statements + the judge
    assert [r["caller"] for r in records] == ["debate_agents._statement"] * 4 + ["debate_agents.run"]
    prompt, completion = totals(records)
    out = capsys.readouterr().out
    assert f"LLM trace (debate): 5 call(s), {prompt:,} prompt + {completion:,} completion tokens" in out
    assert "estimated" in out
    # The debate transcript is re-sent every turn, so prompts only grow.
    statement_prompts = [r["usage"]["prompt_tokens"] for r in records[:4]]
    assert statement_prompts == sorted(statement_prompts) and statement_prompts[0] < statement_prompts[-1]
    first = records[0]
    assert first["system"].startswith("You are the PRO advocate")
    assert first["params"]["temperature"] == 0.6
    assert first["messages"][0]["role"] == "system" and first["response"]["content"]


def test_summary_totals_equal_recorded_cassette_usage(tmp_path, monkeypatch, capsys):
    cassette = json.loads((CASSETTES / "debate.json").read_text(encoding="utf-8"))
    for i, entry in enumerate(cassette["interactions"]):
        entry["usage"] = {"prompt_tokens": 400 + 100 * i, "completion_tokens": 150 + i}
    recorded = tmp_path / "debate.json"
    recorded.write_text(json.dumps(cassette), encoding="utf-8")
    monkeypatch.setattr(main, "cassette_for", lambda pattern: recorded)

    path = tmp_path / "t.jsonl"
    assert main.main(["debate", "--offline", "--trace", str(path)]) == 0
    records = read_jsonl(path)
    assert totals(records) == (400 + 500 + 600 + 700 + 800, 150 + 151 + 152 + 153 + 154)
    assert not any(r["usage"]["estimated"] for r in records)
    out = capsys.readouterr().out
    assert "3,000 prompt + 760 completion tokens" in out
    assert "estimated" not in out.split("LLM trace")[1]


def test_orchestrator_trace_includes_nested_worker_calls(tmp_path):
    path = tmp_path / "orch.jsonl"
    assert main.main(["orchestrator", "--offline", "--trace", str(path)]) == 0
    callers = [r["caller"] for r in read_jsonl(path)]
    assert len(callers) == 10
    assert callers.count("react_agent.run") == 3
    assert callers.count("reflection_agent._critique") == 2
    assert {"orchestrator.run", "orchestrator._direct_worker", "reflection_agent.run"} <= set(callers)


def test_tracing_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main.main(["router", "--offline"]) == 0
    assert nim.find_layer(TracingClient) is None
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_nim_trace_env_var_enables_tracing(tmp_path, monkeypatch):
    path = tmp_path / "env.jsonl"
    monkeypatch.setenv("NIM_TRACE", str(path))
    monkeypatch.setenv("NIM_CASSETTE", "router")
    from src.patterns import router_agent

    router_agent.run(router_agent.DEFAULT_GOAL)
    assert [r["caller"] for r in read_jsonl(path)] == ["router_agent.classify", "router_agent.run"]


def test_method_callers_are_named_by_class(tmp_path):
    nim.configure(backend="mock", cassette="memory", trace_path=nim.MEMORY_TRACE)
    from src.patterns import memory_agent

    memory_agent.run(memory_agent.DEFAULT_GOAL)
    callers = [r["caller"] for r in nim.find_layer(TracingClient).records]
    assert callers == ["MemoryAgent.turn", "memory_agent.extract_facts"] * 3


def test_failed_calls_are_traced_and_re_raised(tmp_path):
    path = tmp_path / "err.jsonl"
    tracer = TracingClient(ReplayClient.scripted(), path)
    with pytest.raises(Exception):
        tracer.chat.completions.create(messages=[{"role": "system", "content": "You are X"}])
    record = read_jsonl(path)[0]
    assert record["error"].startswith("CassetteExhausted") and record["system"] == "You are X"
    assert "1 call(s) raised an error" in tracer.format_summary()


def test_trace_cli_summarizes_files(tmp_path, capsys):
    path = tmp_path / "debate.jsonl"
    main.main(["debate", "--offline", "--trace", str(path)])
    capsys.readouterr()
    assert trace.main([str(path)]) == 0
    assert "5 call(s)" in capsys.readouterr().out
    assert trace.main([]) == 2


def test_token_estimates():
    assert trace.estimate_tokens("") == 0
    assert trace.estimate_tokens("abcd" * 25) == 25
    request = {"messages": [{"role": "user", "content": "a" * 40}],
               "tools": [{"type": "function", "function": {"name": "x"}}]}
    assert trace.estimate_prompt_tokens(request) > 14
