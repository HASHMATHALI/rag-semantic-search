"""End-to-end test against the bundled sample corpus (builds the index if missing)."""
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    models = tmp_path_factory.mktemp("models")
    env = {**os.environ, "MODELS_DIR": str(models)}
    subprocess.run([sys.executable, os.path.join(BACKEND, "build_index.py"), "--out", str(models)], check=True, env=env)
    os.environ["MODELS_DIR"] = str(models)
    os.environ.pop("GROQ_API_KEY", None)
    import main
    with TestClient(main.app) as c:
        yield c


def test_health_and_map(client):
    assert client.get("/health").json() == {"status": "ok", "vectors_in_memory": 36, "answers_enabled": False}
    points = client.get("/map").json()["points"]
    assert len(points) == 36 and {"x", "y", "text"} <= set(points[0])


def test_on_topic_query_is_charted_with_evidence(client):
    body = client.post("/search", json={"query": "artificial intelligence detecting pneumonia in chest X-rays", "top_k": 3}).json()
    top = body["results"][0]
    assert "pneumonia" in top["text"].lower()
    assert set(top["found_by"]) == {"semantic", "lexical"}
    assert top["evidence"] and body["coverage"]["verdict"] == "charted"


def test_off_topic_query_is_uncharted_and_never_answered(client):
    body = client.post("/search", json={"query": "sourdough starter hydration ratio for rye bread", "top_k": 3, "use_rag": True}).json()
    assert body["coverage"]["verdict"] == "uncharted"
    assert body["rag_response"] is None


def test_modes_and_validation(client):
    lexical = client.post("/search", json={"query": "quantum computer", "mode": "lexical", "top_k": 2}).json()
    assert all(r["found_by"] == ["lexical"] for r in lexical["results"])
    assert "quantum" in lexical["results"][0]["text"].lower()
    assert client.post("/search", json={"query": "x", "mode": "psychic"}).status_code == 422
    assert client.post("/search", json={"query": "  "}).status_code == 400
