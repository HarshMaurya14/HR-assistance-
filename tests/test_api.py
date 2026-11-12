import pytest

from api import create_app
from tests.conftest import sql


class EchoLLM:
    """Fake model that records the system prompt it was given."""

    def __init__(self):
        self.system_prompts = []

    def __call__(self, messages, tools=None):
        self.system_prompts.append(messages[0]["content"])
        return {"role": "assistant", "content": "hello from the fake model"}


@pytest.fixture
def llm():
    return EchoLLM()


@pytest.fixture
def app(llm):
    return create_app({"TESTING": True, "LLM": llm})


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, employee_id="EMP001", password="harsh123"):
    return client.post("/login", json={"employee_id": employee_id, "password": password})


def test_pages_require_login(client):
    assert client.get("/").headers["Location"].endswith("/login")
    assert client.post("/chat", json={"query": "hi"}).status_code == 401
    assert client.get("/me").status_code == 401
    assert client.post("/chat/reset").status_code == 401


def test_successful_login_shows_the_app_for_that_user(client):
    response = login(client)
    assert response.status_code == 200 and response.get_json()["employee"]["name"] == "Harsh Maurya"
    assert "password" not in str(response.get_json()).lower()
    page = client.get("/")
    assert page.status_code == 200 and b"Harsh Maurya" in page.data


@pytest.mark.parametrize("employee_id,password", [("EMP001", "wrong"), ("EMP999", "harsh123"), ("", "")])
def test_bad_credentials(client, employee_id, password):
    assert login(client, employee_id, password).status_code == 401


def test_login_is_case_insensitive_on_id(client):
    assert login(client, "emp001").status_code == 200


def test_repeated_failures_lock_the_account_temporarily(client):
    for _ in range(5):
        assert login(client, password="wrong").status_code == 401
    assert login(client, password="harsh123").status_code == 429      # even the right password
    assert login(client, "EMP002", "rahul123").status_code == 200      # other accounts unaffected


def test_chat_uses_the_session_identity_not_the_request_body(client, llm):
    login(client)
    response = client.post("/chat", json={"query": "hi", "employee_id": "EMP003"})
    assert response.status_code == 200
    assert response.get_json() == {"employee_id": "EMP001", "response": "hello from the fake model"}
    assert "Harsh Maurya" in llm.system_prompts[0] and "Priya" not in llm.system_prompts[0]


def test_chat_validates_input(client):
    login(client)
    assert client.post("/chat", json={"query": "   "}).status_code == 400
    assert client.post("/chat", data="not json", content_type="text/plain").status_code == 400


def test_logout_ends_the_session(client):
    login(client)
    assert client.post("/logout").status_code == 200
    assert client.post("/chat", json={"query": "hi"}).status_code == 401


def test_chat_reset_clears_the_conversation(client, app):
    login(client)
    client.post("/chat", json={"query": "hi"})
    store = app.extensions["hr_store"]
    assert store.history("EMP001")
    assert client.post("/chat/reset").status_code == 200
    assert store.history("EMP001") == []


def test_internal_errors_do_not_leak_details(client, app):
    def broken(messages, tools=None):
        raise RuntimeError("secret internal detail")
    app.config["LLM"] = broken
    login(client)
    response = client.post("/chat", json={"query": "hi"})
    assert response.status_code == 500 and "secret" not in response.get_data(as_text=True)


def test_end_to_end_leave_flow_over_http(client, app):
    def apply_then_nothing(messages, tools=None):
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function", "function": {"name": "apply_leave", "arguments":
            '{"leave_type": "Sick Leave", "start_date": "2099-01-05", "end_date": "2099-01-06"}'}}]}
    app.config["LLM"] = apply_then_nothing
    login(client)
    assert "Please confirm" in client.post("/chat", json={"query": "sick leave"}).get_json()["response"]
    assert sql("SELECT * FROM leave_requests") == []
    assert "submitted" in client.post("/chat", json={"query": "yes"}).get_json()["response"]
    assert sql("SELECT employee_id FROM leave_requests") == [{"employee_id": "EMP001"}]


def test_security_headers_and_cookie_flags(client):
    response = login(client)
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_demo_accounts_hidden_outside_demo_mode(client, monkeypatch):
    assert b"harsh123" in client.get("/login").data           # demo mode shows the demo logins
    monkeypatch.setenv("SEED_DEMO_DATA", "0")
    page = client.get("/login").data
    assert b"harsh123" not in page and b"demo-btn\" data-id" not in page


def test_health_endpoint(client):
    assert client.get("/health").get_json() == {"status": "ok"}
