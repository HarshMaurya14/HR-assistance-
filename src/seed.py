"""Create the database and load demo data.

    python -m src.seed

DEMO ONLY: these passwords are public in the README. Never seed a real deployment.
"""

from src.auth import hash_password
from src.database import connect, init_db

# (id, name, phone, department, role, manager_id, demo password)
# Managers are inserted first so the manager_id foreign key is satisfied.
DEMO_EMPLOYEES = [
    ("EMP003", "Priya Mehta", "9876543212", "Management", "Manager", None, "priya123"),
    ("EMP001", "Harsh Maurya", "9876543210", "Engineering", "Employee", "EMP003", "harsh123"),
    ("EMP002", "Rahul Sharma", "9876543211", "Human Resources", "Employee", "EMP003", "rahul123"),
]

# (employee_id, casual, sick, annual) - days remaining
DEMO_BALANCES = [
    ("EMP001", 5, 7, 12),
    ("EMP002", 8, 4, 15),
    ("EMP003", 10, 9, 18),
]


def seed_demo_data() -> None:
    init_db()
    with connect() as connection:
        for emp_id, name, phone, dept, role, manager_id, password in DEMO_EMPLOYEES:
            connection.execute(
                """INSERT OR IGNORE INTO employees
                   (employee_id, name, phone_number, department, role, manager_id, password_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (emp_id, name, phone, dept, role, manager_id, hash_password(password)),
            )
        connection.executemany(
            """INSERT OR IGNORE INTO leave_balances
               (employee_id, casual_leave, sick_leave, annual_leave)
               VALUES (?, ?, ?, ?)""",
            DEMO_BALANCES,
        )


if __name__ == "__main__":
    seed_demo_data()
    print("Database created and demo employees added (EMP001, EMP002, EMP003).")
