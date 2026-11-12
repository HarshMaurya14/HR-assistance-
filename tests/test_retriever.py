import pytest

from src.retriever import retrieve

TOP_1 = [
    ("How do I claim health insurance?", "insurance_policy.md"),
    ("What are the working hours?", "working_hours.md"),
    ("Can I work from home?", "remote_work_policy.md"),
    ("How many sick leaves do I get per year?", "leave_policy.md"),
    ("How much notice do I need for annual leave?", "leave_policy.md"),
    ("Is Saturday a working day?", "working_hours.md"),
    ("Where can I get my payslip?", "payroll_policy.md"),
    ("What does the code of conduct say about harassment?", "code_of_conduct.md"),
    ("What happens if I am repeatedly late?", "attendance_policy.md"),
    ("How do reimbursements work?", "payroll_policy.md"),
    ("What is the company holiday calendar?", "holiday_policy.md"),
]


@pytest.mark.parametrize("question,source", TOP_1)
def test_best_hit_is_the_right_document(question, source):
    assert retrieve(question, top_k=1)[0]["source"] == source


def test_eligibility_question_finds_insurance_policy_in_top_3():
    assert "insurance_policy.md" in [h["source"] for h in retrieve("When am I eligible for health insurance?")]


@pytest.mark.parametrize("question", ["What's the capital of France?", "Tell me a joke about pizza", "", "   "])
def test_unrelated_or_empty_questions_return_nothing(question):
    assert retrieve(question) == []


def test_hits_carry_source_and_section():
    hit = retrieve("sick leave entitlement", top_k=1)[0]
    assert hit["source"] == "leave_policy.md" and hit["section"] == "Sick Leave" and hit["score"] > 0
