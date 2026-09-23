"""Shared fixtures. Every test runs offline, isolated and without network.

- ``no_network`` (autouse): any socket connect or DNS lookup raises, so a
  test that accidentally reaches for the real NIM endpoint fails loudly.
- ``isolated`` (autouse): clears NIM/NVIDIA env vars, defaults the backend
  to ``mock`` (a test that forgets to configure a client can never go live),
  hides the repo's .env, redirects the memory store / audit log / HITL
  workspace into tmp_path, and gives the process a non-interactive stdin.
- ``scripted``: install an in-memory client that answers with the given
  replies, in order, and records every request it received.
"""

from __future__ import annotations

import io
import socket
import sys

import pytest

from src import nim
from src.patterns import human_in_the_loop, memory_agent
from src.replay import ReplayClient
from tests.helpers import NetworkBlocked


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkBlocked("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for var in ("NVIDIA_API_KEY", "NIM_BACKEND", "NIM_CASSETTE", "NIM_RECORD",
                "NIM_TRACE", "NIM_MODEL", "NIM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NIM_BACKEND", "mock")
    monkeypatch.setattr(nim, "ENV_FILE", tmp_path / "absent.env")
    monkeypatch.setattr(memory_agent, "MEMORY_FILE", tmp_path / "memory_store.json")
    monkeypatch.setattr(human_in_the_loop, "AUDIT_LOG", tmp_path / "audit_log.jsonl")
    monkeypatch.setattr(human_in_the_loop, "WORKSPACE", tmp_path / "hitl_workspace")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    nim.reset()
    yield tmp_path
    nim.reset()


@pytest.fixture
def scripted():
    """scripted("reply", {"tool_calls": [...]}, ...) -> the installed client."""

    def install(*replies) -> ReplayClient:
        client = ReplayClient.scripted(*replies)
        nim.configure(backend="live", client=client)
        return client

    return install
