import pytest
import requests

from src import llm


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_request_format_includes_tools(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, payload=json)
        return FakeResponse(200, OK)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    tools = [{"type": "function", "function": {"name": "x"}}]
    assert llm.chat([{"role": "user", "content": "hi"}], tools)["content"] == "hi"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["payload"]["tools"] == tools and sent["payload"]["tool_choice"] == "auto"


def test_retries_once_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    responses = [FakeResponse(429), FakeResponse(200, OK)]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: responses.pop(0))
    assert llm.chat([{"role": "user", "content": "hi"}])["content"] == "hi"


def test_gives_up_with_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    def boom(*a, **k):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(llm.requests, "post", boom)
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_malformed_response_is_an_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(200, {"unexpected": 1}))
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}])
