import pytest

from app.core.exceptions import UnsafeSQLError
from app.db.repositories import assert_safe_select


def test_select_is_allowed():
    assert_safe_select("SELECT * FROM products WHERE id = :id")


def test_insert_is_rejected():
    with pytest.raises(UnsafeSQLError):
        assert_safe_select("INSERT INTO products VALUES (1)")


def test_multiple_statements_rejected():
    with pytest.raises(UnsafeSQLError):
        assert_safe_select("SELECT 1; DROP TABLE products")


def test_drop_keyword_rejected():
    with pytest.raises(UnsafeSQLError):
        assert_safe_select("SELECT * FROM products WHERE name = 'x'; DROP TABLE products")
