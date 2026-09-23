# Changelog

## Unreleased

### Added

- **Offline mode.** `python main.py <pattern> --offline` replays `cassettes/<pattern>.json`: no API key, no network. One hand-written cassette ships for each of the 12 patterns. Tools, parsing, subprocess runs and validation still execute for real. Cassette drift (changed prompts or control flow) fails with a message naming the call and exit code 2.
- **Record mode.** `--record PATH` (or `NIM_BACKEND=record NIM_RECORD=PATH`) runs live and saves every exchange, including request parameters and provider token usage, as a replayable cassette.
- **Tracing.** `--trace FILE.jsonl` (or `NIM_TRACE`) logs every chat call with the calling pattern function, messages, parameters, reply, latency and token usage, and prints a per-caller summary. `python -m src.trace FILE...` summarizes saved traces.
- **`--all --offline`**: runs every pattern and prints a calls/tokens comparison table.
- `NIM_BASE_URL` to point the lab at a self-hosted NIM container.
- Orchestrator: a `debate` worker, next to `react`, `reflection` and `direct`.
- Offline pytest suite (386 tests, network blocked), `requirements-dev.txt`, and a `pyproject.toml` that holds only the pytest configuration.

### Fixed

- All four safe calculators: overflowing and giant results (`1e300**2`, `(9**99)**99`), `inf` and complex results used to raise and kill the agent loop. They now return an error the model can read.
- ReAct: quoted arguments (`convert_units(42, "km", "mi")`) always failed. The greedy `Action:` regex also swallowed text from following lines. Added `Action Input:` support, truncation of self-written observations, and a guard against a `Final Answer` guessed after an `Action`.
- Tool calling: wrong argument types and tool exceptions crashed the loop. They now return a JSON error.
- Human-in-the-loop, orchestrator and memory parsers crashed on bare-string items. The planner rendered dict steps as Python reprs.
- Planner: a replan reply without a `steps` key silently discarded the rest of the plan. The remaining plan is now kept.
- Memory: multi-word keywords (`"property management"`) could never match. Duplicate facts piled up on every run.
- Guardrails: the phone regex required a country code, so `555-123-4567` and `(507) 6123-4567` were not redacted. Any 13–16 digit number was flagged as a card, and a Luhn check now applies. `ignore the previous instructions` and similar phrasings slipped past the screen. The free regex screen now runs before any API client is created, so the injection demo works without a key.
- Code interpreter: generated code could read `NVIDIA_API_KEY` from the inherited environment and import the venv's site-packages, contrary to its docstring. It now runs with `-I -S`, an allowlisted environment and closed stdin, and its temp directory is deleted.
- Structured output: removed the dead `JSONDecodeError` branch; malformed JSON goes through the same Pydantic repair path. Added the subtotal + tax = total cross-check.
- The `nvapi-XXXX...` placeholder from `.env.example` was accepted as a real key and produced a remote 401. It now gets the setup message, and no request is sent.
- Printing model output with arrows or emoji crashed on Windows consoles that use a legacy code page.
- Human-in-the-loop crashed with `EOFError` under `< NUL` on Windows, where the NUL device reports itself as a terminal.

### Changed

- `.env` is loaded only from the repository root; `python-dotenv` no longer searches parent directories.
- The memory demo's default goal plays two scripted follow-up turns when stdin is not a terminal, so a single run shows retrieval. On a terminal it stays interactive. That means up to four extra LLM calls in live, non-interactive runs.
- Human-in-the-loop writes invalid proposals to the audit log as `"decision": "discarded"`, with the reason.
- `guardrails_agent.heuristic_screen` returns the matched phrase instead of the regex source.
- `tool_calling_agent.get_current_time` reports `utc_offset_hours` as a float.
