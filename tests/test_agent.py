"""Agent tests use a scripted fake model, so they are fast, free and deterministic."""

import json

import pytest

from src import agent, hr_tools
from src.agent import ConversationStore, run_hr_agent
from src.llm import LLMError
from tests.conftest import sql


class FakeLLM:
    """Plays back a script of assistant messages and records every call."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise AssertionError("The model was called but no reply was scripted")
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def call(name, args, call_id="call_1"):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": arguments}}]}


def say(text):
    return {"role": "assistant", "content": text}


def ask(query, employee_id, llm, store, today):
    return run_hr_agent(query, employee_id, llm=llm, store=store, today=today)


@pytest.fixture
def store():
    return ConversationStore()


def tool_names(tools):
    return {t["function"]["name"] for t in tools}


# ------------------------------------------------------------ read-only tools
def test_balance_question_uses_tool_result(store, today):
    llm = FakeLLM(call("get_leave_balance", {}), say("You have 5 casual days."))
    assert ask("how many leaves do I have?", "EMP001", llm, store, today) == "You have 5 casual days."
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and '"casual_leave": 5' in tool_message["content"]


def test_policy_question_returns_sources_to_the_model(store, today):
    llm = FakeLLM(call("search_policies", {"query": "how do I claim health insurance"}),
                  say("Inform HR first. (Source: insurance_policy.md)"))
    ask("how do I claim insurance?", "EMP001", llm, store, today)
    payload = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert payload["results"][0]["source"] == "insurance_policy.md"


def test_system_prompt_knows_user_and_date(store, today):
    llm = FakeLLM(say("hi"))
    ask("hello", "EMP001", llm, store, today)
    system = llm.calls[0]["messages"][0]["content"]
    assert "Harsh Maurya" in system and "2026-11-09" in system and "Monday" in system


def test_follow_up_messages_include_history(store, today):
    llm = FakeLLM(say("Which dates?"), say("Okay"))
    ask("I need sick leave", "EMP001", llm, store, today)
    ask("next Tuesday to Thursday", "EMP001", llm, store, today)
    second = llm.calls[1]["messages"]
    assert [m["content"] for m in second[1:3]] == ["I need sick leave", "Which dates?"]


# --------------------------------------------------- confirmation flow (writes)
APPLY = {"leave_type": "Sick Leave", "start_date": "2026-11-10", "end_date": "2026-11-12", "reason": "fever"}


def test_leave_requires_confirmation_before_saving(store, today):
    llm = FakeLLM(call("apply_leave", APPLY))
    reply = ask("sick leave Tue-Thu", "EMP001", llm, store, today)
    assert "Please confirm" in reply and "3 working days" in reply and "**yes**" in reply
    assert sql("SELECT * FROM leave_requests") == []          # nothing saved yet

    reply = ask("yes", "EMP001", llm, store, today)            # no new model call is scripted
    assert "#1" in reply and "Pending" in reply
    row = sql("SELECT employee_id, status, days FROM leave_requests")[0]
    assert row == {"employee_id": "EMP001", "status": "Pending", "days": 3}


def test_saying_no_cancels(store, today):
    llm = FakeLLM(call("apply_leave", APPLY))
    ask("sick leave Tue-Thu", "EMP001", llm, store, today)
    assert "Nothing was changed" in ask("no", "EMP001", llm, store, today)
    assert sql("SELECT * FROM leave_requests") == []
    assert ask("yes", "EMP001", FakeLLM(say("Nothing pending.")), store, today) == "Nothing pending."


def test_changing_the_subject_drops_the_pending_action(store, today):
    llm = FakeLLM(call("apply_leave", APPLY), say("The working hours are 9 to 6."), say("ok"))
    ask("sick leave Tue-Thu", "EMP001", llm, store, today)
    ask("actually, what are the working hours?", "EMP001", llm, store, today)
    ask("yes", "EMP001", llm, store, today)                    # goes to the model, not to the DB
    assert sql("SELECT * FROM leave_requests") == []


def test_expired_confirmation_is_not_executed(today):
    store = ConversationStore(pending_ttl=0)
    llm = FakeLLM(call("apply_leave", APPLY))
    ask("sick leave Tue-Thu", "EMP001", llm, store, today)
    assert "expired" in ask("yes", "EMP001", llm, store, today)
    assert sql("SELECT * FROM leave_requests") == []


def test_invalid_leave_is_explained_not_proposed(store, today):
    too_long = {"leave_type": "Casual Leave", "start_date": "2026-11-10", "end_date": "2026-11-20"}
    llm = FakeLLM(call("apply_leave", too_long), say("You only have 5 casual days."))
    reply = ask("casual leave for 9 days", "EMP001", llm, store, today)
    assert reply == "You only have 5 casual days."
    assert "only" in llm.calls[1]["messages"][-1]["content"]   # the validation error reached the model
    assert not store.has_pending("EMP001")


def test_model_cannot_choose_whose_leave_it_is(store, today):
    sneaky = {**APPLY, "employee_id": "EMP003", "employeeId": "EMP003"}
    llm = FakeLLM(call("apply_leave", sneaky))
    ask("sick leave Tue-Thu", "EMP001", llm, store, today)
    ask("yes", "EMP001", llm, store, today)
    assert sql("SELECT employee_id FROM leave_requests") == [{"employee_id": "EMP001"}]


def test_cancel_flow(store, today):
    request_id = hr_tools.apply_leave("EMP001", "Casual Leave", "2026-11-10", "2026-11-10", today=today)["request_id"]
    llm = FakeLLM(call("cancel_leave", {"request_id": request_id}))
    assert "Please confirm" in ask("cancel my request", "EMP001", llm, store, today)
    assert "Cancelled" in ask("yes", "EMP001", llm, store, today)
    assert sql("SELECT status FROM leave_requests")[0]["status"] == "Cancelled"


# ----------------------------------------------------------- manager permissions
def test_employees_are_not_offered_manager_tools():
    employee_tools, manager_tools = tool_names(agent.tools_for("Employee")), tool_names(agent.tools_for("Manager"))
    assert not employee_tools & agent.MANAGER_ONLY
    assert agent.MANAGER_ONLY <= manager_tools


def test_tool_list_sent_to_the_model_depends_on_role(store, today):
    employee_llm, manager_llm = FakeLLM(say("hi")), FakeLLM(say("hi"))
    ask("hi", "EMP001", employee_llm, store, today)
    ask("hi", "EMP003", manager_llm, store, today)
    assert "approve_leave" not in tool_names(employee_llm.calls[0]["tools"])
    assert "approve_leave" in tool_names(manager_llm.calls[0]["tools"])


def test_hallucinated_manager_tool_is_refused_for_employees(store, today):
    request_id = hr_tools.apply_leave("EMP002", "Casual Leave", "2026-11-10", "2026-11-10", today=today)["request_id"]
    llm = FakeLLM(call("approve_leave", {"request_id": request_id}), say("Sorry, only managers can do that."))
    ask(f"approve request {request_id}", "EMP001", llm, store, today)
    assert "isn't available" in llm.calls[1]["messages"][-1]["content"]
    assert not store.has_pending("EMP001")
    assert sql("SELECT status FROM leave_requests")[0]["status"] == "Pending"


def test_manager_can_review_and_approve(store, today):
    request_id = hr_tools.apply_leave("EMP001", "Casual Leave", "2026-11-10", "2026-11-11", today=today)["request_id"]

    llm = FakeLLM(call("list_pending_approvals", {}), say("Harsh has one pending request."))
    ask("what's waiting for me?", "EMP003", llm, store, today)
    listed = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert listed["requests"][0]["request_id"] == request_id

    llm = FakeLLM(call("approve_leave", {"request_id": request_id}))
    assert "Please confirm" in ask(f"approve request {request_id}", "EMP003", llm, store, today)
    reply = ask("yes", "EMP003", llm, store, today)
    assert "Approved" in reply and "**3**" in reply            # 5 - 2 days remaining
    assert hr_tools.get_leave_balance("EMP001")["casual_leave"] == 3


def test_manager_cannot_approve_unknown_request(store, today):
    llm = FakeLLM(call("approve_leave", {"request_id": 99}), say("I couldn't find that request."))
    ask("approve request 99", "EMP003", llm, store, today)
    assert not store.has_pending("EMP003")


# ------------------------------------------------------------------ robustness
def test_llm_outage_gives_a_friendly_message(store, today):
    reply = ask("hello", "EMP001", FakeLLM(LLMError("boom")), store, today)
    assert "trouble reaching" in reply and "boom" not in reply


def test_malformed_tool_arguments_do_not_crash(store, today):
    llm = FakeLLM(call("apply_leave", "{not json"), say("Could you share the dates?"))
    assert ask("sick leave", "EMP001", llm, store, today) == "Could you share the dates?"


def test_runaway_tool_loop_is_stopped(store, today):
    llm = FakeLLM(*[call("get_leave_balance", {}, f"c{i}") for i in range(agent.MAX_STEPS)])
    assert "wasn't able to finish" in ask("balance", "EMP001", llm, store, today)


def test_unknown_employee_and_empty_query(store, today):
    assert "couldn't identify" in ask("hi", "NOBODY", FakeLLM(), store, today)
    assert "Please type" in ask("   ", "EMP001", FakeLLM(), store, today)


def test_reset_clears_history_and_pending(store, today):
    llm = FakeLLM(call("apply_leave", APPLY))
    ask("sick leave Tue-Thu", "EMP001", llm, store, today)
    agent.reset_conversation("EMP001", store)
    assert not store.has_pending("EMP001") and store.history("EMP001") == []
