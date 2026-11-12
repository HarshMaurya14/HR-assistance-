"""HR business logic.

All rules live here, not in the LLM layer:

* identity is always passed in by the server (never by the model),
* every state-changing function can run as a dry run (``commit=False``),
  which validates everything and returns a human-readable summary,
* writes run in one SQLite transaction so checks and updates cannot race.
"""

from datetime import date, datetime, timedelta

from src import config
from src.database import connect

LEAVE_TYPES = {
    "Casual Leave": "casual_leave",
    "Sick Leave": "sick_leave",
    "Annual Leave": "annual_leave",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fail(message: str) -> dict:
    return {"ok": False, "message": message}


def fmt_date(value: date) -> str:
    return value.strftime("%a %d %b %Y")


def fmt_span(start: date, end: date) -> str:
    return fmt_date(start) if start == end else f"{fmt_date(start)} to {fmt_date(end)}"


def _span_from_iso(start_iso: str, end_iso: str) -> str:
    return fmt_span(date.fromisoformat(start_iso), date.fromisoformat(end_iso))


def normalize_leave_type(value) -> str | None:
    """Accept 'sick', 'Sick leave', 'SICK LEAVE' ... and return the canonical name."""
    if not value:
        return None
    cleaned = str(value).strip().lower().replace("leave", "").strip()
    for canonical in LEAVE_TYPES:
        if canonical.lower().replace(" leave", "") == cleaned:
            return canonical
    return None


def parse_date(value) -> date:
    """Parse YYYY-MM-DD. Raises ValueError with a friendly message."""
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        raise ValueError(f"'{value}' is not a valid date. Please use YYYY-MM-DD.") from None


def count_working_days(start: date, end: date, holidays: dict | None = None) -> int:
    """Mon-Fri days in [start, end] that are not company holidays."""
    holidays = config.load_holidays() if holidays is None else holidays
    total, current = 0, start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            total += 1
        current += timedelta(days=1)
    return total


# ---------------------------------------------------------------------------
# Employees and balances
# ---------------------------------------------------------------------------

def get_employee(employee_id: str) -> dict | None:
    with connect() as connection:
        row = connection.execute(
            """SELECT employee_id, name, department, role, manager_id
               FROM employees WHERE employee_id = ?""",
            (employee_id,),
        ).fetchone()
    return dict(row) if row else None


def get_employee_role(employee_id: str) -> str | None:
    employee = get_employee(employee_id)
    return employee["role"] if employee else None


def _pending_days(connection, employee_id: str, leave_type: str) -> int:
    return connection.execute(
        """SELECT COALESCE(SUM(days), 0) FROM leave_requests
           WHERE employee_id = ? AND leave_type = ? AND status = 'Pending'""",
        (employee_id, leave_type),
    ).fetchone()[0]


def get_leave_balance(employee_id: str) -> dict | None:
    """Remaining days per type, plus days already tied up in pending requests."""
    with connect() as connection:
        row = connection.execute(
            """SELECT casual_leave, sick_leave, annual_leave
               FROM leave_balances WHERE employee_id = ?""",
            (employee_id,),
        ).fetchone()
        if row is None:
            return None
        pending = {lt: _pending_days(connection, employee_id, lt) for lt in LEAVE_TYPES}

    balance = dict(row)
    balance["pending_days"] = pending
    return balance


def check_leave_balance(employee_id: str, leave_type: str) -> int | None:
    balance = get_leave_balance(employee_id)
    canonical = normalize_leave_type(leave_type)
    if balance is None or canonical is None:
        return None
    return balance[LEAVE_TYPES[canonical]]


# ---------------------------------------------------------------------------
# Apply for leave
# ---------------------------------------------------------------------------

def _validate_leave(connection, employee_id, leave_type, start_date, end_date, today) -> dict:
    canonical = normalize_leave_type(leave_type)
    if canonical is None:
        return _fail("Leave type must be Casual Leave, Sick Leave or Annual Leave.")

    try:
        start, end = parse_date(start_date), parse_date(end_date)
    except ValueError as error:
        return _fail(str(error))

    if end < start:
        return _fail("The end date can't be before the start date.")
    if (end - start).days + 1 > config.MAX_LEAVE_SPAN_DAYS:
        return _fail(f"A single request can't span more than {config.MAX_LEAVE_SPAN_DAYS} days.")

    earliest = today - timedelta(days=config.SICK_LEAVE_BACKDATE_DAYS) \
        if canonical == "Sick Leave" else today
    if start < earliest:
        if canonical == "Sick Leave":
            return _fail(
                f"Sick leave can be reported up to {config.SICK_LEAVE_BACKDATE_DAYS} days "
                "after the fact. Please contact HR for older dates."
            )
        return _fail("Leave can't start in the past. Please choose a future date.")

    days = count_working_days(start, end)
    if days == 0:
        return _fail("That range only covers weekends or company holidays, so no leave is needed.")

    overlap = connection.execute(
        """SELECT request_id, start_date, end_date, status FROM leave_requests
           WHERE employee_id = ? AND status IN ('Pending', 'Approved')
             AND start_date <= ? AND end_date >= ?""",
        (employee_id, end.isoformat(), start.isoformat()),
    ).fetchone()
    if overlap:
        return _fail(
            f"These dates overlap with your {overlap['status'].lower()} request "
            f"#{overlap['request_id']} ({overlap['start_date']} to {overlap['end_date']})."
        )

    balance_row = connection.execute(
        f"SELECT {LEAVE_TYPES[canonical]} FROM leave_balances WHERE employee_id = ?",  # column from a fixed whitelist
        (employee_id,),
    ).fetchone()
    if balance_row is None:
        return _fail("I couldn't find your leave balance.")

    balance = balance_row[0]
    pending = _pending_days(connection, employee_id, canonical)
    available = balance - pending
    if days > available:
        return _fail(
            f"You asked for {days} working day(s) of {canonical}, but only {available} "
            f"are available ({balance} remaining, {pending} already in pending requests)."
        )

    warnings = []
    lead_days = (start - today).days
    if canonical == "Annual Leave" and lead_days < config.ANNUAL_LEAVE_NOTICE_DAYS:
        warnings.append(
            f"Policy asks for {config.ANNUAL_LEAVE_NOTICE_DAYS} days' notice for annual leave, "
            "so your manager may decline."
        )
    if canonical == "Casual Leave" and lead_days < config.CASUAL_LEAVE_NOTICE_DAYS:
        warnings.append("Policy asks for casual leave to be requested at least 1 day in advance.")

    span = fmt_span(start, end)
    return {
        "ok": True,
        "leave_type": canonical,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": days,
        "available": available,
        "warnings": warnings,
        "summary": f"{canonical}, {span} ({days} working day{'s' if days != 1 else ''})",
    }


def apply_leave(employee_id: str, leave_type: str, start_date: str, end_date: str,
                reason: str | None = None, *, commit: bool = True,
                today: date | None = None) -> dict:
    """Validate and (optionally) submit a leave request."""
    today = today or date.today()
    reason = (reason or "").strip()[:300] or None

    with connect() as connection:
        if commit:
            connection.execute("BEGIN IMMEDIATE")  # lock before checking, so checks can't go stale

        check = _validate_leave(connection, employee_id, leave_type, start_date, end_date, today)
        if not check["ok"] or not commit:
            return check

        cursor = connection.execute(
            """INSERT INTO leave_requests
               (employee_id, leave_type, start_date, end_date, days, reason, status)
               VALUES (?, ?, ?, ?, ?, ?, 'Pending')""",
            (employee_id, check["leave_type"], check["start_date"], check["end_date"],
             check["days"], reason),
        )
        check["request_id"] = cursor.lastrowid
        check["status"] = "Pending"
        return check


# ---------------------------------------------------------------------------
# Viewing requests
# ---------------------------------------------------------------------------

_REQUEST_COLUMNS = """r.request_id, r.employee_id, e.name AS employee_name, r.leave_type,
                      r.start_date, r.end_date, r.days, r.reason, r.status"""


def list_leave_requests(employee_id: str, limit: int = 20) -> list[dict]:
    """The employee's own requests, newest first."""
    with connect() as connection:
        rows = connection.execute(
            f"""SELECT {_REQUEST_COLUMNS}
                FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id
                WHERE r.employee_id = ?
                ORDER BY r.request_id DESC LIMIT ?""",
            (employee_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def list_pending_approvals(manager_id: str) -> dict:
    """Pending requests from the manager's direct reports."""
    if get_employee_role(manager_id) != "Manager":
        return _fail("Only managers can view pending approvals.")

    with connect() as connection:
        rows = connection.execute(
            f"""SELECT {_REQUEST_COLUMNS}
                FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id
                WHERE e.manager_id = ? AND r.status = 'Pending'
                ORDER BY r.start_date""",
            (manager_id,),
        ).fetchall()
    return {"ok": True, "requests": [dict(row) for row in rows]}


# ---------------------------------------------------------------------------
# Approve / reject (managers only, direct reports only)
# ---------------------------------------------------------------------------

def decide_leave(approver_id: str, request_id: int, approve: bool, *, commit: bool = True) -> dict:
    """Approve or reject a pending request.

    Rules: the approver must be a Manager, must be the requester's manager,
    and can never decide their own request.
    """
    verb = "approve" if approve else "reject"
    try:
        request_id = int(request_id)
    except (TypeError, ValueError):
        return _fail("Please give me a valid leave request number.")

    with connect() as connection:
        if commit:
            connection.execute("BEGIN IMMEDIATE")

        approver = connection.execute(
            "SELECT role FROM employees WHERE employee_id = ?", (approver_id,)
        ).fetchone()
        if approver is None or approver["role"] != "Manager":
            return _fail(f"You are not authorized to {verb} leave requests.")

        request = connection.execute(
            """SELECT r.*, e.name AS employee_name, e.manager_id
               FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id
               WHERE r.request_id = ?""",
            (request_id,),
        ).fetchone()

        # Same message for "doesn't exist" and "not your report": don't leak other teams' data.
        not_found = _fail(f"I couldn't find leave request #{request_id} among your team's requests.")
        if request is None:
            return not_found
        if request["employee_id"] == approver_id:
            return _fail(f"You can't {verb} your own leave request.")
        if request["manager_id"] != approver_id:
            return not_found
        if request["status"] != "Pending":
            return _fail(f"Request #{request_id} is already {request['status'].lower()}.")

        column = LEAVE_TYPES[request["leave_type"]]
        summary = (f"#{request_id} from {request['employee_name']}: {request['leave_type']}, "
                   f"{_span_from_iso(request['start_date'], request['end_date'])} "
                   f"({request['days']} working day{'s' if request['days'] != 1 else ''})")
        result = {"ok": True, "request_id": request_id, "summary": summary,
                  "employee_name": request["employee_name"], "days": request["days"],
                  "leave_type": request["leave_type"]}

        balance = connection.execute(
            f"SELECT {column} FROM leave_balances WHERE employee_id = ?",  # whitelisted column
            (request["employee_id"],),
        ).fetchone()
        if approve:
            if balance is None or balance[0] < request["days"]:
                return _fail(f"{request['employee_name']} no longer has enough "
                             f"{request['leave_type']} balance for this request.")
            result["remaining_balance"] = balance[0] - request["days"]

        if not commit:
            return result

        updated = connection.execute(
            """UPDATE leave_requests
               SET status = ?, decided_by = ?, decided_at = datetime('now')
               WHERE request_id = ? AND status = 'Pending'""",
            ("Approved" if approve else "Rejected", approver_id, request_id),
        ).rowcount
        if updated != 1:
            return _fail(f"Request #{request_id} was already processed.")

        if approve:
            connection.execute(
                f"UPDATE leave_balances SET {column} = {column} - ? WHERE employee_id = ?",
                (request["days"], request["employee_id"]),
            )
        result["status"] = "Approved" if approve else "Rejected"
        return result


def approve_leave(approver_id: str, request_id: int, *, commit: bool = True) -> dict:
    return decide_leave(approver_id, request_id, True, commit=commit)


def reject_leave(approver_id: str, request_id: int, *, commit: bool = True) -> dict:
    return decide_leave(approver_id, request_id, False, commit=commit)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

def cancel_leave(employee_id: str, request_id: int, *, commit: bool = True) -> dict:
    """Employees can cancel their own *pending* requests."""
    try:
        request_id = int(request_id)
    except (TypeError, ValueError):
        return _fail("Please give me a valid leave request number.")

    with connect() as connection:
        if commit:
            connection.execute("BEGIN IMMEDIATE")

        request = connection.execute(
            """SELECT request_id, leave_type, start_date, end_date, days, status
               FROM leave_requests WHERE request_id = ? AND employee_id = ?""",
            (request_id, employee_id),
        ).fetchone()
        if request is None:
            return _fail(f"I couldn't find leave request #{request_id} among your requests.")
        if request["status"] == "Approved":
            return _fail("Approved requests can't be cancelled here. Please contact your manager or HR.")
        if request["status"] != "Pending":
            return _fail(f"Request #{request_id} is already {request['status'].lower()}.")

        summary = (f"#{request_id}: {request['leave_type']}, "
                   f"{_span_from_iso(request['start_date'], request['end_date'])} "
                   f"({request['days']} working day{'s' if request['days'] != 1 else ''})")
        if not commit:
            return {"ok": True, "request_id": request_id, "summary": summary}

        connection.execute(
            "UPDATE leave_requests SET status = 'Cancelled', decided_at = datetime('now') "
            "WHERE request_id = ? AND status = 'Pending'",
            (request_id,),
        )
        return {"ok": True, "request_id": request_id, "summary": summary, "status": "Cancelled"}
