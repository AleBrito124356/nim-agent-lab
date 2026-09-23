from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src import nim
from src.patterns import structured_output_agent as so
from src.replay import CassetteMismatch, ReplayClient, build_response

GOOD = {
    "vendor": "Cordillera Cloud Services", "customer": "Acme Logistics S.A.", "invoice_number": "INV-2093",
    "issue_date": "2026-06-14", "currency": "USD",
    "line_items": [
        {"description": "Managed Kubernetes (June)", "quantity": 1, "unit_price": 890.0},
        {"description": "Object storage, 4 TB", "quantity": 4, "unit_price": 23.5},
        {"description": "Support retainer", "quantity": 1, "unit_price": 350.0},
    ],
    "subtotal": 1334.0, "tax": 93.38, "total": 1427.38,
}


def test_invoice_schema_accepts_good_data():
    assert so.Invoice.model_validate(GOOD).total == 1427.38


@pytest.mark.parametrize(
    "patch",
    [{"issue_date": "14/06/2026"}, {"currency": "usd"}, {"line_items": []}, {"tax": -1}],
)
def test_invoice_schema_rejects_bad_fields(patch):
    with pytest.raises(ValidationError):
        so.Invoice.model_validate({**GOOD, **patch})


def test_cross_checks():
    assert all(ok for _, ok in so.cross_check(so.Invoice.model_validate(GOOD)))
    bad_total = so.Invoice.model_validate({**GOOD, "total": 1500.0})
    checks = dict((text.split(",")[0], ok) for text, ok in so.cross_check(bad_total))
    assert checks["line items sum to 1334.00"] is True
    assert checks["subtotal + tax = 1427.38"] is False


def test_repair_loop_feeds_errors_back(scripted, capsys):
    bad = {**GOOD, "currency": "usd"}
    client = scripted(json.dumps(bad), "```json\n" + json.dumps(GOOD) + "\n```")
    invoice = so.extract(so.SAMPLE_INVOICE)
    assert invoice.currency == "USD"
    repair_prompt = client.requests[1]["messages"][-1]["content"]
    assert '"field": "currency"' in repair_prompt
    assert "[repair] succeeded on repair round 1" in capsys.readouterr().out


def test_malformed_json_goes_through_the_same_repair_path(scripted):
    # audit: the old 'except json.JSONDecodeError' branch was dead code
    client = scripted("{'vendor': 'single quotes are not JSON'", json.dumps(GOOD))
    assert so.extract("doc").vendor == GOOD["vendor"]
    assert "Invalid JSON" in client.requests[1]["messages"][-1]["content"]


def test_extraction_gives_up_after_max_repairs(scripted):
    scripted(*["not json"] * (1 + so.MAX_REPAIRS))
    assert so.run("doc").startswith("Extraction failed: extraction failed after 2 repairs")


def test_run_reports_arithmetic_mismatch(scripted, capsys):
    scripted(json.dumps({**GOOD, "total": 1500.0}))
    so.run(so.SAMPLE_INVOICE)
    out = capsys.readouterr().out
    assert "subtotal + tax = 1427.38, stated total 1500.00 -> MISMATCH" in out


class NoJsonModeClient:
    """Rejects response_format like models without json_object support."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise TypeError("response_format not supported by this model")
        return build_response(self.replies.pop(0), None, {"prompt_tokens": 1, "completion_tokens": 1})


def test_json_mode_fallback_is_remembered(capsys):
    client = NoJsonModeClient(["not json", json.dumps(GOOD)])
    nim.configure(backend="live", client=client)
    so.extract("doc")
    with_format = [c for c in client.calls if "response_format" in c]
    assert len(with_format) == 1 and len(client.calls) == 3   # one probe, then plain only
    assert "json_object mode unavailable" in capsys.readouterr().out


def test_backend_errors_are_not_swallowed_by_the_fallback():
    client = ReplayClient({"interactions": [{"expect": "something else", "response": {"content": "{}"}}]})
    nim.configure(backend="live", client=client)
    with pytest.raises(CassetteMismatch):
        so.extract("doc")
