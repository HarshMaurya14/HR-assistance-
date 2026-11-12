"""Flask web app: login, session handling and the /chat endpoint."""

import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from src import agent, config
from src.auth import verify_login
from src.database import employee_count, init_db
from src.hr_tools import get_employee
from src.seed import seed_demo_data

log = logging.getLogger("hr_agent")

MAX_LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 300


class LoginLimiter:
    """Blocks repeated failed logins for the same (IP, employee ID)."""

    def __init__(self, max_failures=MAX_LOGIN_FAILURES, window=LOGIN_WINDOW_SECONDS):
        self._failures = defaultdict(deque)
        self._lock = threading.Lock()
        self._max, self._window = max_failures, window

    def _recent(self, key):
        queue = self._failures[key]
        cutoff = time.monotonic() - self._window
        while queue and queue[0] < cutoff:
            queue.popleft()
        return queue

    def is_blocked(self, key) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self._max

    def record_failure(self, key) -> None:
        with self._lock:
            self._recent(key).append(time.monotonic())

    def reset(self, key) -> None:
        with self._lock:
            self._failures.pop(key, None)


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)

    secret = os.getenv("SECRET_KEY")
    if not secret:
        secret = secrets.token_hex(32)
        log.warning("SECRET_KEY is not set: using a random key, so logins reset on restart. "
                    "Set SECRET_KEY in .env.")
    app.config.update(
        SECRET_KEY=secret,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "0") == "1",  # set 1 behind HTTPS
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        LLM=None,  # tests inject a fake model here
    )
    if test_config:
        app.config.update(test_config)

    init_db()
    if config.demo_mode() and employee_count() == 0:
        seed_demo_data()

    store = agent.ConversationStore()
    limiter = LoginLimiter()
    app.extensions["hr_store"] = store
    app.extensions["login_limiter"] = limiter

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    def current_employee():
        employee_id = session.get("employee_id")
        return get_employee(employee_id) if employee_id else None

    # ----- pages ------------------------------------------------------------
    @app.get("/")
    def home():
        employee = current_employee()
        if employee is None:
            session.clear()
            return redirect(url_for("login_page"))
        return render_template("index.html", employee_id=employee["employee_id"],
                               employee_name=employee["name"])

    @app.get("/login")
    def login_page():
        if current_employee():
            return redirect(url_for("home"))
        return render_template("login.html", demo_mode=config.demo_mode())

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # ----- auth API -----------------------------------------------------------
    @app.post("/login")
    def login():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Please send your login details as JSON."}), 400

        employee_id = str(data.get("employee_id", "")).strip().upper()
        password = str(data.get("password", ""))
        key = (request.remote_addr, employee_id)

        if limiter.is_blocked(key):
            return jsonify({"error": "Too many failed attempts. Please wait a few minutes."}), 429

        employee = verify_login(employee_id, password)
        if employee is None:
            limiter.record_failure(key)
            return jsonify({"error": "Invalid Employee ID or password"}), 401

        limiter.reset(key)
        session.clear()                     # new session on login (prevents fixation)
        session.permanent = True
        session["employee_id"] = employee["employee_id"]
        return jsonify({"message": "Login successful", "employee": employee}), 200

    @app.get("/me")
    def me():
        employee = current_employee()
        if employee is None:
            session.clear()
            return jsonify({"error": "Not logged in"}), 401
        return jsonify(employee), 200

    @app.post("/logout")
    def logout():
        employee_id = session.get("employee_id")
        if employee_id:
            agent.reset_conversation(employee_id, store)
        session.clear()
        return jsonify({"message": "Logged out successfully"}), 200

    # ----- chat API -----------------------------------------------------------
    @app.post("/chat")
    def chat():
        employee = current_employee()
        if employee is None:
            return jsonify({"error": "Please log in first"}), 401

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "No JSON data provided"}), 400
        query = str(data.get("query", "")).strip()
        if not query:
            return jsonify({"error": "Please enter a question"}), 400

        try:
            # The employee ID comes from the signed session, never from the request body.
            response = agent.run_hr_agent(
                query, employee["employee_id"], llm=app.config["LLM"], store=store
            )
        except Exception:
            log.exception("Unhandled error while processing chat message")
            return jsonify({"error": "Something went wrong while processing your request."}), 500

        return jsonify({"employee_id": employee["employee_id"], "response": response}), 200

    @app.post("/chat/reset")
    def chat_reset():
        employee = current_employee()
        if employee is None:
            return jsonify({"error": "Please log in first"}), 401
        agent.reset_conversation(employee["employee_id"], store)
        return jsonify({"message": "Conversation cleared"}), 200

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
