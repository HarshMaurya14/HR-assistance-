"""Password hashing and login verification."""

from werkzeug.security import check_password_hash, generate_password_hash

from src.database import connect

# Used so that "unknown user" and "wrong password" take the same time.
_DUMMY_HASH = generate_password_hash("not-a-real-password")


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_login(employee_id: str, password: str) -> dict | None:
    """Return the employee (without secrets) if the credentials are valid."""
    with connect() as connection:
        row = connection.execute(
            """SELECT employee_id, name, department, role, manager_id, password_hash
               FROM employees WHERE employee_id = ?""",
            (employee_id,),
        ).fetchone()

    stored_hash = row["password_hash"] if row and row["password_hash"] else _DUMMY_HASH
    password_ok = check_password_hash(stored_hash, password)

    if row is None or not row["password_hash"] or not password_ok:
        return None

    return {
        "employee_id": row["employee_id"],
        "name": row["name"],
        "department": row["department"],
        "role": row["role"],
        "manager_id": row["manager_id"],
    }
