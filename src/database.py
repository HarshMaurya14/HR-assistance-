"""SQLite access: one schema, one place to create it."""

import sqlite3
from contextlib import contextmanager

from src.config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    phone_number  TEXT UNIQUE,
    department    TEXT,
    role          TEXT NOT NULL DEFAULT 'Employee'
                  CHECK (role IN ('Employee', 'Manager')),
    manager_id    TEXT REFERENCES employees(employee_id),
    password_hash TEXT
);

CREATE TABLE IF NOT EXISTS leave_balances (
    employee_id   TEXT PRIMARY KEY REFERENCES employees(employee_id),
    casual_leave  INTEGER NOT NULL DEFAULT 0 CHECK (casual_leave >= 0),
    sick_leave    INTEGER NOT NULL DEFAULT 0 CHECK (sick_leave >= 0),
    annual_leave  INTEGER NOT NULL DEFAULT 0 CHECK (annual_leave >= 0)
);

CREATE TABLE IF NOT EXISTS leave_requests (
    request_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    leave_type  TEXT NOT NULL
                CHECK (leave_type IN ('Casual Leave', 'Sick Leave', 'Annual Leave')),
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    days        INTEGER NOT NULL CHECK (days > 0),   -- working days, fixed at request time
    reason      TEXT,
    status      TEXT NOT NULL DEFAULT 'Pending'
                CHECK (status IN ('Pending', 'Approved', 'Rejected', 'Cancelled')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    decided_by  TEXT REFERENCES employees(employee_id),
    decided_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_leave_requests_employee
    ON leave_requests (employee_id, status);
"""


def get_connection() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def connect():
    """Commit on success, roll back on error, always close."""
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create all tables (safe to call repeatedly)."""
    with connect() as connection:
        connection.executescript(SCHEMA)


def employee_count() -> int:
    with connect() as connection:
        return connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
