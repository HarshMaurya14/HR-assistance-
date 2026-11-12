import shutil
from datetime import date

import pytest

from src.database import get_connection
from src.seed import seed_demo_data

TODAY = date(2026, 11, 9)  # a Monday


@pytest.fixture(scope="session")
def seeded_template(tmp_path_factory):
    """Seed once (password hashing is slow on purpose) and copy per test."""
    path = tmp_path_factory.mktemp("template") / "template.db"
    patch = pytest.MonkeyPatch()
    patch.setenv("HR_DB_PATH", str(path))
    seed_demo_data()
    patch.undo()
    return path


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch, seeded_template):
    """Fresh, seeded database for every test."""
    target = tmp_path / "test.db"
    shutil.copy(seeded_template, target)
    monkeypatch.setenv("HR_DB_PATH", str(target))
    monkeypatch.setenv("SECRET_KEY", "test-secret")


@pytest.fixture
def today():
    return TODAY


def sql(query: str, params=()):
    """Tiny helper for assertions against the database."""
    connection = get_connection()
    try:
        rows = connection.execute(query, params).fetchall()
        connection.commit()
        return [dict(r) for r in rows]
    finally:
        connection.close()
