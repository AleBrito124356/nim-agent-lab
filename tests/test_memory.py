from __future__ import annotations

import json

import pytest

from src.patterns import memory_agent as mem

FACTS = [
    {"fact": "User builds a property management app", "keywords": ["property management", "fastapi"]},
    {"fact": "User prefers Postgres over MySQL", "keywords": ["postgres", "mysql", "database"]},
    {"fact": "User lives in Panama City", "keywords": ["panama city"]},
]


def test_multi_word_keywords_match():
    # audit: 'property management' could never match single-word query tokens
    assert mem.retrieve_facts(FACTS, "any tips for my property management project?") == [
        "User builds a property management app"
    ]


def test_retrieval_ranks_by_overlap_and_respects_k():
    query = "postgres or mysql database for my fastapi property app"
    assert mem.retrieve_facts(FACTS, query)[0] == "User prefers Postgres over MySQL"
    assert len(mem.retrieve_facts(FACTS, query, k=1)) == 1


def test_no_overlap_retrieves_nothing():
    assert mem.retrieve_facts(FACTS, "what's the weather like?") == []


def test_facts_without_keywords_fall_back_to_their_own_words():
    assert mem.retrieve_facts([{"fact": "User owns a golden retriever"}], "my retriever is sick") == [
        "User owns a golden retriever"
    ]


def test_parse_facts_accepts_model_variants():
    reply = json.dumps({"facts": [
        "User deploys on a single VPS",
        {"text": "User's deadline is March 3", "keywords": "deadline, march"},
        {"fact": "User likes tea"},
        42,
    ]})
    facts = mem.parse_facts(reply, known=[])
    assert [f["fact"] for f in facts] == [
        "User deploys on a single VPS", "User's deadline is March 3", "User likes tea"
    ]
    assert facts[1]["keywords"] == ["deadline", "march"]
    assert "vps" in facts[0]["keywords"]


def test_parse_facts_deduplicates_against_known_and_within_batch():
    known = [{"fact": "User prefers Postgres over MySQL.", "keywords": []}]
    reply = json.dumps({"facts": ["user prefers postgres over mysql", "New fact", "new fact!"]})
    assert [f["fact"] for f in mem.parse_facts(reply, known)] == ["New fact"]


@pytest.mark.parametrize("reply", ["no json", '{"facts": "none"}', '{"other": []}', '{"facts": null}'])
def test_parse_facts_garbage_is_empty(reply):
    assert mem.parse_facts(reply, []) == []


def test_store_roundtrip_and_atomic_write():
    mem.save_facts(FACTS)
    assert mem.load_facts() == FACTS
    assert not mem.MEMORY_FILE.with_name(mem.MEMORY_FILE.name + ".tmp").exists()


def test_load_facts_skips_malformed_entries():
    mem.MEMORY_FILE.write_text(json.dumps(["just a string", {"no": "fact"}, {"fact": "ok", "keywords": "a, b"}]),
                               encoding="utf-8")
    assert mem.load_facts() == [{"fact": "ok", "keywords": ["a", "b"]}]


@pytest.mark.parametrize("content", ["{not json", '{"a": 1}', ""])
def test_load_facts_tolerates_corrupt_store(content):
    mem.MEMORY_FILE.write_text(content, encoding="utf-8")
    assert mem.load_facts() == []


def test_window_overflow_is_compressed_into_summary(scripted, monkeypatch, capsys):
    monkeypatch.setattr(mem, "SHORT_TERM_MAX_MESSAGES", 2)
    client = scripted(
        "hello!", '{"facts": []}',                              # turn 1
        "sure.", "User said hi; assistant greeted.", '{"facts": []}',   # turn 2 + compression
    )
    agent = mem.MemoryAgent()
    agent.turn("hi")
    agent.turn("help me")
    assert agent.summary == "User said hi; assistant greeted."
    assert len(agent.window) == 2
    assert client.position == 5
    assert "compressed 2 old message(s)" in capsys.readouterr().out


def test_retrieved_facts_reach_the_system_prompt(scripted):
    mem.save_facts(FACTS)
    client = scripted("Use Postgres.", '{"facts": []}')
    mem.MemoryAgent().turn("Which database should I pick?")
    system = client.requests[0]["messages"][0]["content"]
    assert "User prefers Postgres over MySQL" in system
    assert "property management" not in system
