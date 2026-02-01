import os
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_system.app.api.query import router as query_router
from rag_system.app.observability.ratelimit import rate_limiter


class DummyStore:
    def search(self, query_vector, k=10, user_id=None, doc_id=None):
        return [
            {
                "content": "hello world",
                "user_id": "user1",
                "doc_id": "doc1",
                "source": "doc.pdf",
                "chunk_index": 0,
            },
            {
                "content": "more text",
                "user_id": "user1",
                "doc_id": "doc1",
                "source": "doc.pdf",
                "chunk_index": 1,
            },
        ][:k]


class DummyOpenAI:
    class _Chat:
        class _Completions:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )

        completions = _Completions()

    chat = _Chat()


def test_query_endpoint_basic(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "test"

    app = FastAPI()
    app.include_router(query_router, prefix="/api")
    app.state.langfuse = None

    monkeypatch.setattr(
        "rag_system.app.api.query.get_store",
        lambda: DummyStore(),
    )
    monkeypatch.setattr(
        "rag_system.app.api.query.OpenAI",
        lambda api_key=None: DummyOpenAI(),
    )
    monkeypatch.setattr(
        "rag_system.app.api.query.embed_texts",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        "rag_system.app.api.query.rerank_with_scores",
        lambda q, docs: [(d, 1.0) for d in docs],
    )
    rate_limiter._events.clear()

    client = TestClient(app)
    resp = client.post(
        "/api/query",
        json={
            "query": "test",
            "user_id": "user1",
            "k": 2,
            "max_context_tokens": 200,
            "max_answer_tokens": 100,
            "temperature": 0.2,
            "rerank": True,
            "include_citations": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Answer"
    assert len(data["results"]) == 2
