"""Production-readiness guard (validate_runtime_config).

It must flag insecure / wrong-environment settings as CRITICAL (block boot) and
likely-wrong ones as warnings — and pass cleanly on a well-formed prod config."""
import types

from app.config import validate_runtime_config


def _cfg(**over):
    base = dict(
        meta_app_secret="app-secret",
        whatsapp_verify_token="vtok",
        meta_access_token="atok",
        public_base_url="https://bot.example.com",
        razorpay_enabled=True,
        razorpay_test_mode=False,
        database_url="postgresql+asyncpg://u:p@prod-db:5432/job7",
        llm_enabled=True,
        llm_api_key="llm-key",
        register_in_db=True,
        employer_register_in_db=True,
        job_post_in_db=True,
        credits_purchase_in_db=True,
        payments_in_db=True,
        subscriptions_in_db=True,
        saved_jobs_in_db=True,
        multilang_enabled=True,
        employer_kyc_auto_verify=False,
        qdrant_api_key="qkey",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_clean_prod_config_has_no_criticals():
    criticals, warnings = validate_runtime_config(_cfg())
    assert criticals == []
    assert warnings == []


def test_missing_app_secret_is_critical():
    criticals, _ = validate_runtime_config(_cfg(meta_app_secret=""))
    assert any("META_APP_SECRET" in c for c in criticals)


def test_test_database_is_critical():
    criticals, _ = validate_runtime_config(
        _cfg(database_url="postgresql+asyncpg://u:p@host/jobs7uat"))
    assert any("jobs7uat" in c for c in criticals)


def test_razorpay_test_mode_is_critical():
    criticals, _ = validate_runtime_config(_cfg(razorpay_test_mode=True))
    assert any("TEST mode" in c for c in criticals)


def test_non_https_base_url_is_critical():
    criticals, _ = validate_runtime_config(_cfg(public_base_url="http://localhost:8000"))
    assert any("PUBLIC_BASE_URL" in c for c in criticals)


def test_persistence_flags_off_are_warnings_not_criticals():
    criticals, warnings = validate_runtime_config(
        _cfg(payments_in_db=False, job_post_in_db=False))
    assert criticals == []
    assert any("PAYMENTS_IN_DB" in w and "JOB_POST_IN_DB" in w for w in warnings)


def test_auto_kyc_is_a_warning():
    _, warnings = validate_runtime_config(_cfg(employer_kyc_auto_verify=True))
    assert any("KYC" in w or "auto-approved" in w for w in warnings)


def test_dev_defaults_trip_multiple_criticals():
    """The current local defaults (no app secret, localhost, test keys, test DB)
    would be refused in production — exactly the misconfigs the guard catches."""
    criticals, _ = validate_runtime_config(_cfg(
        meta_app_secret="", whatsapp_verify_token="", meta_access_token="",
        public_base_url="http://localhost:8000", razorpay_test_mode=True,
        database_url="postgresql://u:p@host/jobs7uat",
    ))
    assert len(criticals) >= 5
