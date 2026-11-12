from datetime import date

import pytest

from src import hr_tools as tools
from tests.conftest import TODAY, sql


# ---------------------------------------------------------------- working days
def test_working_days_skips_weekends():
    assert tools.count_working_days(date(2026, 11, 9), date(2026, 11, 13)) == 5   # Mon-Fri
    assert tools.count_working_days(date(2026, 11, 13), date(2026, 11, 16)) == 2  # Fri-Mon
    assert tools.count_working_days(date(2026, 11, 14), date(2026, 11, 15)) == 0  # Sat-Sun


def test_working_days_skips_company_holidays():
    # Thu 24 Dec .. Mon 28 Dec: 25 Dec is a holiday, 26/27 is a weekend
    assert tools.count_working_days(date(2026, 12, 24), date(2026, 12, 28)) == 2


@pytest.mark.parametrize("raw,expected", [
    ("sick", "Sick Leave"), ("Sick Leave", "Sick Leave"), ("CASUAL LEAVE", "Casual Leave"),
    ("annual", "Annual Leave"), ("holiday", None), ("", None), (None, None),
])
def test_normalize_leave_type(raw, expected):
    assert tools.normalize_leave_type(raw) == expected


# ------------------------------------------------------------------ applying
def test_dry_run_does_not_write(today):
    result = tools.apply_leave("EMP001", "Sick Leave", "2026-11-10", "2026-11-12",
                               commit=False, today=today)
    assert result["ok"] and result["days"] == 3
    assert sql("SELECT * FROM leave_requests") == []


def test_apply_creates_pending_request(today):
    result = tools.apply_leave("EMP001", "sick", "2026-11-10", "2026-11-12", "fever", today=today)
    assert result["ok"] and result["request_id"] == 1 and result["status"] == "Pending"
    row = sql("SELECT * FROM leave_requests")[0]
    assert (row["employee_id"], row["days"], row["status"], row["reason"]) == ("EMP001", 3, "Pending", "fever")
    # Balance is only deducted on approval
    assert tools.get_leave_balance("EMP001")["sick_leave"] == 7


@pytest.mark.parametrize("args,fragment", [
    (("Sick Leave", "2026-11-14", "2026-11-15"), "weekends"),
    (("Casual Leave", "2026-11-02", "2026-11-03"), "past"),
    (("Sick Leave", "2026-10-01", "2026-10-02"), "up to 3 days"),
    (("Casual Leave", "2026-11-12", "2026-11-10"), "before the start"),
    (("Casual Leave", "10/11/2026", "2026-11-12"), "YYYY-MM-DD"),
    (("Parental Leave", "2026-11-10", "2026-11-10"), "Casual Leave, Sick Leave or Annual"),
    (("Annual Leave", "2026-11-10", "2027-03-10"), "more than"),
])
def test_invalid_requests_are_rejected(today, args, fragment):
    result = tools.apply_leave("EMP001", *args, today=today)
    assert not result["ok"] and fragment in result["message"]
    assert sql("SELECT * FROM leave_requests") == []


def test_sick_leave_can_be_reported_a_day_late(today):
    assert tools.apply_leave("EMP001", "Sick Leave", "2026-11-08", "2026-11-09", today=today)["ok"]


def test_overlapping_requests_are_blocked(today):
    assert tools.apply_leave("EMP001", "Casual Leave", "2026-11-10", "2026-11-12", today=today)["ok"]
    clash = tools.apply_leave("EMP001", "Sick Leave", "2026-11-12", "2026-11-13", today=today)
    assert not clash["ok"] and "overlap" in clash["message"]


def test_pending_requests_count_against_balance(today):
    # EMP001 has 5 casual days
    assert tools.apply_leave("EMP001", "Casual Leave", "2026-11-10", "2026-11-13", today=today)["ok"]  # 4 days
    second = tools.apply_leave("EMP001", "Casual Leave", "2026-11-16", "2026-11-17", today=today)    # +2 > 5
    assert not second["ok"] and "only 1 are available" in second["message"]


def test_balance_exceeded(today):
    result = tools.apply_leave("EMP001", "Casual Leave", "2026-11-10", "2026-11-17", today=today)  # 6 days
    assert not result["ok"] and "6 working day" in result["message"]


def test_short_notice_annual_leave_gets_a_warning(today):
    result = tools.apply_leave("EMP001", "Annual Leave", "2026-11-10", "2026-11-10",
                               commit=False, today=today)
    assert result["ok"] and any("notice" in w for w in result["warnings"])


# ------------------------------------------------------------ approve / reject
def _request(today, employee="EMP001", leave_type="Casual Leave", start="2026-11-10", end="2026-11-11"):
    return tools.apply_leave(employee, leave_type, start, end, today=today)["request_id"]


def test_manager_approves_and_balance_is_deducted(today):
    request_id = _request(today)
    result = tools.approve_leave("EMP003", request_id)
    assert result["ok"] and result["remaining_balance"] == 3
    assert tools.get_leave_balance("EMP001")["casual_leave"] == 3
    row = sql("SELECT status, decided_by, decided_at FROM leave_requests")[0]
    assert row["status"] == "Approved" and row["decided_by"] == "EMP003" and row["decided_at"]


def test_cannot_decide_twice(today):
    request_id = _request(today)
    assert tools.approve_leave("EMP003", request_id)["ok"]
    again = tools.approve_leave("EMP003", request_id)
    assert not again["ok"] and "already approved" in again["message"]
    assert tools.get_leave_balance("EMP001")["casual_leave"] == 3  # not deducted twice


def test_reject_does_not_touch_balance(today):
    request_id = _request(today)
    assert tools.reject_leave("EMP003", request_id)["ok"]
    assert tools.get_leave_balance("EMP001")["casual_leave"] == 5
    assert sql("SELECT status FROM leave_requests")[0]["status"] == "Rejected"


def test_employees_cannot_approve(today):
    request_id = _request(today)
    result = tools.approve_leave("EMP002", request_id)  # a colleague, not a manager
    assert not result["ok"] and "not authorized" in result["message"]
    assert sql("SELECT status FROM leave_requests")[0]["status"] == "Pending"


def test_manager_cannot_approve_own_request(today):
    request_id = _request(today, employee="EMP003")
    result = tools.approve_leave("EMP003", request_id)
    assert not result["ok"] and "your own" in result["message"]


def test_manager_cannot_decide_other_teams_requests(today):
    sql("INSERT INTO employees (employee_id, name, role) VALUES ('EMP010', 'Other Manager', 'Manager')")
    sql("INSERT INTO employees (employee_id, name, role, manager_id) VALUES ('EMP011', 'Outsider', 'Employee', 'EMP010')")
    sql("INSERT INTO leave_balances VALUES ('EMP011', 5, 5, 5)")
    request_id = _request(today, employee="EMP011")
    result = tools.approve_leave("EMP003", request_id)
    assert not result["ok"] and "couldn't find" in result["message"]
    assert tools.approve_leave("EMP003", 999)["message"] == result["message"].replace(str(request_id), "999")


def test_balance_is_rechecked_at_approval(today):
    request_id = _request(today)                      # 2 days
    sql("UPDATE leave_balances SET casual_leave = 1 WHERE employee_id = 'EMP001'")
    result = tools.approve_leave("EMP003", request_id)
    assert not result["ok"] and "no longer has enough" in result["message"]


def test_pending_approvals_only_show_direct_reports(today):
    _request(today)
    _request(today, employee="EMP002", start="2026-11-12", end="2026-11-12")
    pending = tools.list_pending_approvals("EMP003")
    assert pending["ok"] and {r["employee_id"] for r in pending["requests"]} == {"EMP001", "EMP002"}
    assert not tools.list_pending_approvals("EMP001")["ok"]


# ------------------------------------------------------------------- cancel
def test_cancel_own_pending_request(today):
    request_id = _request(today)
    assert tools.cancel_leave("EMP001", request_id)["ok"]
    assert sql("SELECT status FROM leave_requests")[0]["status"] == "Cancelled"
    # the dates are free again
    assert tools.apply_leave("EMP001", "Casual Leave", "2026-11-10", "2026-11-11", today=today)["ok"]


def test_cannot_cancel_someone_elses_or_approved_request(today):
    request_id = _request(today)
    assert not tools.cancel_leave("EMP002", request_id)["ok"]
    tools.approve_leave("EMP003", request_id)
    result = tools.cancel_leave("EMP001", request_id)
    assert not result["ok"] and "Approved requests" in result["message"]
