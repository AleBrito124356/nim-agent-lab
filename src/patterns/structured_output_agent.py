"""Structured output: Pydantic schema in, validated object out, repairs on failure.

The flow that actually survives production:

1. Derive the JSON schema from a Pydantic model (single source of truth).
2. Ask for JSON, preferring the endpoint's native json_object response
   format when the model supports it (graceful fallback when it does not).
3. Validate with Pydantic -- NOT just json.loads. Types, required fields
   and value constraints all get checked.
4. On ValidationError, feed the exact error list back to the model and ask
   for a corrected object. Two repair rounds recover almost every failure.

Run standalone:
    python -m src.patterns.structured_output_agent
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from src.nim import get_client, get_model  # noqa: E402

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


def _strip_to_json(reply: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", reply).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0) if match else cleaned


def _request_json(client, messages: list[dict]) -> str:
    """Ask for JSON; try native json_object mode first, fall back to plain."""
    model = get_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Not every NIM model supports response_format; plain prompting
        # plus validation-repair below covers the difference.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1200,
        )
    return (response.choices[0].message.content or "").strip()


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

    last_error = ""
    for attempt in range(1 + MAX_REPAIRS):
        raw = _request_json(client, messages)
        candidate = _strip_to_json(raw)
        try:
            invoice = Invoice.model_validate_json(candidate)
            if attempt:
                print(f"[repair] succeeded on repair round {attempt}")
            return invoice
        except ValidationError as exc:
            errors = [
                {"field": ".".join(str(loc) for loc in e["loc"]), "problem": e["msg"]}
                for e in exc.errors()
            ]
            last_error = json.dumps(errors, indent=2)
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
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"That was not parseable JSON ({exc}). Output ONLY the JSON object.",
                }
            )

    raise ValueError(f"extraction failed after {MAX_REPAIRS} repairs: {last_error}")


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

    # A cheap cross-check validation alone cannot express:
    computed = round(sum(li.quantity * li.unit_price for li in invoice.line_items), 2)
    flag = "matches" if abs(computed - invoice.subtotal) < 0.01 else "MISMATCH"
    print(f"[check] line items sum to {computed}, stated subtotal {invoice.subtotal} -> {flag}")
    print("-" * 72)
    return result


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip()
    run(goal or SAMPLE_INVOICE)
