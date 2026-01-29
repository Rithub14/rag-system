from enterprise_rag_system.app.response.context_builder import build_context


def test_build_context_limits_and_returns_used_chunks():
    chunks = [
        {"source": "a.txt", "chunk_index": 0, "content": "A" * 10},
        {"source": "a.txt", "chunk_index": 1, "content": "B" * 10},
        {"source": "a.txt", "chunk_index": 2, "content": "C" * 10},
    ]
    context, used = build_context("q", chunks, max_tokens=60)
    assert "[a.txt#0]" in context
    assert "[a.txt#1]" in context
    assert "[a.txt#2]" not in context
    assert len(used) == 2
