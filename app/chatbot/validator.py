"""Anti-hallucination validation layer.

The LLM is constrained by the system prompt, but we still verify its output
against the retrieved context. We check:

- every salary figure mentioned (₹NNN, Rs NNN, INR NNN) exists in the job rows
- every job/role title mentioned exists in retrieved data
- no job references (JOB-XXXX) or application references (APP-XXXX) are quoted
  that weren't returned by a tool
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    offending: list[str] | None = None


# Salary figures the model might quote. Also matches "25L"/"25 lakh" loosely via
# the bare-number path in validate(); the currency-prefixed form is checked here.
_PRICE_RX = re.compile(
    r"(?:₹|rs\.?|inr)\s*([0-9]+(?:[,\.][0-9]{2,3})*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
# Human-facing job + application references. The separator is REQUIRED (refs
# are always "JOB-AB1001" / "APP-CD5678") so plain English words like "job",
# "apply" or "application" aren't mistaken for a reference. A digit in the
# suffix keeps it from matching an all-letters word after the dash.
_JOB_RX = re.compile(r"\bJOB[-_](?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b", re.IGNORECASE)
_APP_RX = re.compile(r"\bAPP[-_](?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b", re.IGNORECASE)

# Indian salary shorthand the model writes for large figures: "₹15 LPA",
# "₹15 lakh", "₹15L", "₹30k", "₹1.5 cr". The stored salary_min/max are raw
# integers (e.g. 1_500_000), so we must scale the quoted number by its unit
# before grounding it — otherwise "₹15 LPA" reads as the literal 15 and a
# perfectly grounded salary is wrongly flagged as a hallucination. Longer units
# are listed first so the regex matches "lakh" before the bare "l", etc.
_SALARY_UNIT_RX = re.compile(
    r"\s*(crores?|cr|lakhs?|lacs?|lpa|l|k)\b", re.IGNORECASE
)
_UNIT_SCALE = {
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "lpa": 100_000, "l": 100_000,
    "k": 1_000,
}


def _salary_unit_scale(trailing: str) -> Decimal | None:
    """Return the multiplier for a salary unit immediately following a quoted
    figure (e.g. " LPA" → 100000), or None when no unit is present."""
    m = _SALARY_UNIT_RX.match(trailing)
    if not m:
        return None
    return Decimal(_UNIT_SCALE[m.group(1).lower()])


def _normalise_price(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def _allowed_prices(sql_rows: Iterable[dict[str, Any]]) -> set[Decimal]:
    out: set[Decimal] = set()
    for r in sql_rows:
        for key in ("salary_min", "salary_max", "salary", "price"):
            v = r.get(key)
            if v is None:
                continue
            try:
                out.add(Decimal(str(v)))
            except Exception:  # noqa: BLE001
                continue
    return out


def _allowed_names(sql_rows: Iterable[dict[str, Any]], vector_hits: Iterable[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for r in sql_rows:
        for key in ("title", "job_title", "name"):
            if r.get(key):
                names.add(str(r[key]).lower())
    for h in vector_hits:
        md = h.get("metadata") or {}
        for key in ("title", "name"):
            if md.get(key):
                names.add(str(md[key]).lower())
    return names


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    """Walk nested tool results — get_application_status returns
    {"application": {...}} / {"applications": [...]} — so refs one level deep
    still count as grounding."""
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _iter_dicts(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_dicts(v)


def _allowed_job_refs(sql_rows: Iterable[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for top in sql_rows:
        for r in _iter_dicts(top):
            if r.get("job_ref"):
                out.add(str(r["job_ref"]).lower())
    return out


def _allowed_app_refs(sql_rows: Iterable[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for top in sql_rows:
        for r in _iter_dicts(top):
            for key in ("app_ref", "application_ref"):
                if r.get(key):
                    out.add(str(r[key]).lower())
    return out


class HallucinationValidator:
    def validate(
        self,
        response: str,
        *,
        sql_rows: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        customer_query: str | None = None,
    ) -> ValidationResult:
        offending: list[str] = []

        # Allowed salary figures: those returned by the job rows plus any number
        # the candidate already supplied (e.g. their expected salary). Echoing
        # the candidate's own number back to them is not a hallucination.
        prices = _allowed_prices(sql_rows)
        if customer_query:
            for m in re.finditer(r"\b(\d+(?:[,\.]\d{2,3})*(?:\.\d+)?)\b", customer_query):
                try:
                    prices.add(_normalise_price(m.group(1)))
                except Exception:  # noqa: BLE001
                    continue
        for m in _PRICE_RX.finditer(response):
            try:
                amount = _normalise_price(m.group(1))
            except Exception:  # noqa: BLE001
                continue
            # "₹15 LPA" / "₹15 lakh" / "₹30k" → scale to the real figure so the
            # shorthand of a grounded salary (1_500_000) isn't read as a bare 15.
            scale = _salary_unit_scale(response[m.end():m.end() + 8])
            if scale is not None:
                amount *= scale
            if amount not in prices:
                offending.append(f"unsupported_salary:{amount}")

        job_allowed = _allowed_job_refs(sql_rows)
        for m in _JOB_RX.finditer(response):
            if m.group(0).lower() not in job_allowed:
                offending.append(f"unsupported_job_ref:{m.group(0)}")

        app_allowed = _allowed_app_refs(sql_rows)
        for m in _APP_RX.finditer(response):
            if m.group(0).lower() not in app_allowed:
                offending.append(f"unsupported_app_ref:{m.group(0)}")

        # Role title check: if a quoted "Senior X Engineer"-style phrase doesn't
        # appear in retrieved data, flag it. Heuristic, low recall by design.
        allowed_names = _allowed_names(sql_rows, vector_hits)
        for quoted in re.findall(r"\"([^\"]{3,80})\"", response):
            normalised = quoted.lower().strip(" \t\n,.!?;:'\"-")
            if not normalised:
                continue
            if normalised in allowed_names:
                continue
            if any(normalised in n or n in normalised for n in allowed_names):
                continue
            offending.append(f"unsupported_name:{quoted}")

        if offending:
            return ValidationResult(
                valid=False,
                reason="grounding_failed",
                offending=offending,
            )
        return ValidationResult(valid=True)
