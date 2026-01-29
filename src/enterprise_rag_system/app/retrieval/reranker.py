from typing import Dict, List, Tuple

from .embeddings import embed_texts


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return _dot(a, a) ** 0.5


def rerank_with_scores(query: str, docs: List[Dict]) -> List[Tuple[Dict, float]]:
    if not docs:
        return []

    query_emb = embed_texts([query])[0]
    doc_embs = embed_texts([doc.get("content", "") for doc in docs])
    qn = _norm(query_emb) or 1.0

    scores = []
    for i, emb in enumerate(doc_embs):
        dn = _norm(emb) or 1.0
        score = _dot(query_emb, emb) / (qn * dn)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [(docs[i], score) for i, score in scores]


def rerank(query: str, docs: List[Dict]) -> List[Dict]:
    return [doc for doc, _ in rerank_with_scores(query, docs)]
