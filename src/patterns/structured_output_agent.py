"""Structured output: Pydantic schema in, validated object out, repairs on failure.

The flow that actually survives production:

1. Derive the JSON schema from a Pydantic model (single source of truth).
2. Ask for JSON, preferring the endpoint's native json_object response
   format when the model supports it (graceful fallback when it does not,
   remembered for the rest of the extraction so repairs do not pay twice).
3. Validate with Pydantic -- NOT just json.loads. Types, required fields,
   value constraints and malformed JSON (Pydantic's ``json_invalid`` error)
   all come back as one ValidationError.
4. On ValidationError, feed the exact error list back to the model and ask
   for a corrected object. Two repair rounds recover almost every failure.
5. Cross-check what a schema cannot express: line items must add up to the
   subtotal, and subtotal + tax must equal the total. Mismatches are
   reported, not silently "fixed" -- the document itself may be wrong.

Run standalone:
    python -m src.patterns.structured_output_agent
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from src.nim import BackendError, extract_json, get_client, get_model  # noqa: E402

DESCRIPTION = "Pydantic schema -> JSON extraction with a validation-error repair loop."

SAMPLE_INVOICE = """\
INVOICE  no INV-2093 -- issued 2026-06-14
From: Cordillera Cloud Services, Panama City
Bill to: Acme Logistics S.A.

  * Managed Kubernetes (June)  ......  1 x 890.00
  * Object storage, 4 TB  ...........  4 x 23.50
  * Support retainer  ...............  1 x 350.00

subtotal 1334.00 / ITBMS 7% 93.38 / TOTAL USD 1427.38
Payment due within 30 days.
"""

DEFAULT_GOAL = SAMPLE_INVOICE
MAX_REPAIRS = 2
MONEY_TOLERANCE = 0.01


class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)


class Invoice(BaseModel):
    vendor: str = Field(min_length=1)
    customer: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    issue_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date YYYY-MM-DD")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(ge=0)


def _candidate_json(reply: str) -> str:
    """The JSON object inside the reply, re-serialized; or the raw reply, so
    Pydantic reports exactly why it is not JSON (error type json_invalid)."""
    data = extract_json(reply)
    return json.dumps(data) if data is not None else reply


def _request_json(client, messages: list[dict], state: dict) -> str:
    """Ask for JSON; try native json_object mode first, fall back to plain."""
    model = get_model()
    if state.get("json_mode", True):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            return (response.choices[0].message.content or "").strip()
        except BackendError:
            raise
        except Exception as exc:
            # Not every NIM model supports response_format; plain prompting
            # plus validation-repair below covers the difference.
            print(f"[structured-output] json_object mode unavailable ({type(exc).__name__}); "
                  "using plain prompting")
            state["json_mode"] = False
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1200,
    )
    return (response.choices[0].message.content or "").strip()


def format_errors(exc: ValidationError) -> str:
    """The error list the model sees: field path + problem, nothing else."""
    errors = [
        {"field": ".".join(str(loc) for loc in e["loc"]) or "(whole object)", "problem": e["msg"]}
        for e in exc.errors()
    ]
    return json.dumps(errors, indent=2)


def extract(document: str) -> Invoice:
    """Extract an Invoice from free text; raises ValueError if repairs fail."""
    client = get_client()
    schema = json.dumps(Invoice.model_json_schema(), indent=2)
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "Extract the invoice data from the document into JSON matching "
                f"this JSON Schema exactly. Output ONLY the JSON object.\n\n{schema}"
            ),
        },
        {"role": "user", "content": document},
    ]

    state: dict = {}
    last_error = ""
    for attempt in range(1 + MAX_REPAIRS):
        raw = _request_json(client, messages, state)
        try:
            invoice = Invoice.model_validate_json(_candidate_json(raw))
        except ValidationError as exc:
            last_error = format_errors(exc)
            print(f"[validate] attempt {attempt + 1} failed:\n{last_error}")
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The JSON failed validation with these errors:\n"
                        f"{last_error}\n\nFix every listed field and output the "
                        "COMPLETE corrected JSON object only."
                    ),
                }
            )
            continue
        if attempt:
            print(f"[repair] succeeded on repair round {attempt}")
        return invoice

    raise ValueError(f"extraction failed after {MAX_REPAIRS} repairs: {last_error}")


def cross_check(invoice: Invoice) -> list[tuple[str, bool]]:
    """Arithmetic a schema cannot express. Returns (description, ok) pairs."""
    items_sum = round(sum(li.quantity * li.unit_price for li in invoice.line_items), 2)
    expected_total = round(invoice.subtotal + invoice.tax, 2)
    return [
        (
            f"line items sum to {items_sum:.2f}, stated subtotal {invoice.subtotal:.2f}",
            abs(items_sum - invoice.subtotal) < MONEY_TOLERANCE,
        ),
        (
            f"subtotal + tax = {expected_total:.2f}, stated total {invoice.total:.2f}",
            abs(expected_total - invoice.total) < MONEY_TOLERANCE,
        ),
    ]


def run(goal: str) -> str:
    document = goal if goal.strip() else SAMPLE_INVOICE
    print("\n[structured-output] extracting invoice fields...\n" + "-" * 72)
    try:
        invoice = extract(document)
    except ValueError as exc:
        return f"Extraction failed: {exc}"

    result = invoice.model_dump_json(indent=2)
    print("[structured-output] validated object:")
    print(result)

    for description, ok in cross_check(invoice):
        print(f"[check] {description} -> {'matches' if ok else 'MISMATCH'}")
    print("-" * 72)
    return result


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip()
    run(goal or SAMPLE_INVOICE)
