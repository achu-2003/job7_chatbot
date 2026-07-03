"""Structured-log PII redaction (_scrub / _redact_pii).

Phone numbers, names, and secrets must be masked before a record is rendered —
keyed by field name so unrelated numbers (amounts, counts) are left intact, and
recursing into the raw Meta webhook payload."""
from app.core.logging import _mask_phone, _redact_pii, _scrub


def test_phone_value_masked_keeps_last4():
    out = _scrub({"customer_number": "919042177457"})
    masked = out["customer_number"]
    assert masked.endswith("7457") and "*" in masked
    assert "9042177457" not in masked


def test_secrets_and_tokens_fully_redacted():
    out = _scrub({
        "razorpay_key_secret": "abcd1234",
        "access_token": "EAAB...long",
        "authorization": "Bearer x",
        "onboard_token": "tok123",
    })
    assert out["razorpay_key_secret"] == "***"
    assert out["access_token"] == "***"
    assert out["authorization"] == "***"
    assert out["onboard_token"] == "***"


def test_app_name_fields_masked():
    out = _scrub({"full_name": "Prasanth", "company_name": "Egleminds"})
    assert out["full_name"] == "P***"
    assert out["company_name"] == "E***"


def test_non_pii_numbers_preserved():
    out = _scrub({"amount": 1500, "count": 3, "latency_ms": 1011, "ref": "JOB-42"})
    assert out == {"amount": 1500, "count": 3, "latency_ms": 1011, "ref": "JOB-42"}


def test_bare_name_outside_payload_not_masked():
    # a plan/intent/logger "name" is not PII and must stay readable
    out = _scrub({"name": "product_search", "intent": "browse"})
    assert out["name"] == "product_search"


def test_nested_webhook_payload_is_scrubbed():
    payload = {
        "payload": {
            "entry": [{
                "changes": [{
                    "value": {
                        "contacts": [{"wa_id": "919042177457",
                                      "profile": {"name": "Asha Kumar"}}],
                        "messages": [{"from": "919042177457", "id": "wamid.X"}],
                        "metadata": {"display_phone_number": "918000000000"},
                    }
                }]
            }]
        }
    }
    out = _redact_pii(None, "info", payload)
    value = out["payload"]["entry"][0]["changes"][0]["value"]
    assert value["contacts"][0]["wa_id"].endswith("7457")
    assert "9042177457" not in value["contacts"][0]["wa_id"]
    assert value["contacts"][0]["profile"]["name"] == "A***"   # contact name (PII here)
    assert value["messages"][0]["from"].endswith("7457")
    assert value["messages"][0]["id"] == "wamid.X"             # non-PII id intact
    assert value["metadata"]["display_phone_number"].endswith("0000")


def test_mask_phone_short_value_is_blanked():
    assert _mask_phone("123") == "***"
