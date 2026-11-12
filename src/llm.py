"""Thin client for Groq's OpenAI-compatible chat API (with tool calling)."""

import time

import requests

from src import config

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(Exception):
    """The language model could not be reached or returned something unusable."""


def chat(messages: list[dict], tools: list[dict] | None = None, *,
         temperature: float = 0.1, max_tokens: int = 800) -> dict:
    """Send a chat request and return the assistant message dict.

    The returned dict has ``content`` and, if the model wants tools, ``tool_calls``.
    """
    api_key = config.groq_api_key()
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    payload = {
        "model": config.groq_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    last_error = None
    for attempt in range(2):  # one retry for transient failures
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30,
            )
            if response.status_code in (429, 500, 502, 503) and attempt == 0:
                time.sleep(1.5)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            last_error = error
            if attempt == 0:
                time.sleep(1.0)

    raise LLMError(f"LLM request failed: {last_error}")
