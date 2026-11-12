"""Command-line chat, handy for demos and quick testing.

    python main.py EMP001
"""

import sys

from src import config
from src.agent import run_hr_agent
from src.database import employee_count, init_db
from src.hr_tools import get_employee
from src.seed import seed_demo_data


def main() -> None:
    employee_id = (sys.argv[1] if len(sys.argv) > 1 else "EMP001").upper()

    init_db()
    if config.demo_mode() and employee_count() == 0:
        seed_demo_data()

    employee = get_employee(employee_id)
    if employee is None:
        sys.exit(f"Unknown employee ID: {employee_id}")

    print(f"HR Assistant - signed in as {employee['name']} ({employee['role']}). Type 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if query:
            print(f"\nHR Assistant: {run_hr_agent(query, employee_id)}\n")


if __name__ == "__main__":
    main()
