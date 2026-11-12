from src.auth import verify_login
from tests.conftest import sql


def test_valid_login_returns_employee_without_secrets():
    employee = verify_login("EMP001", "harsh123")
    assert employee["name"] == "Harsh Maurya" and employee["role"] == "Employee"
    assert "password_hash" not in employee and "password" not in employee


def test_wrong_password_and_unknown_user_fail():
    assert verify_login("EMP001", "wrong") is None
    assert verify_login("NOPE", "harsh123") is None
    assert verify_login("EMP001", "") is None


def test_passwords_are_stored_hashed():
    stored = sql("SELECT password_hash FROM employees")
    assert stored and all(r["password_hash"] and "123" not in r["password_hash"] for r in stored)


def test_employee_without_password_cannot_log_in():
    sql("UPDATE employees SET password_hash = NULL WHERE employee_id = 'EMP001'")
    assert verify_login("EMP001", "harsh123") is None
