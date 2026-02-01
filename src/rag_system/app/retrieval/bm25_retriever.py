from typing import List, Tuple

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, docs: List[str]) -> None:
        self.docs = docs
        tokenized = [doc.split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)

    def get_top_n(self, query: str, n: int = 5) -> List[int]:
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:n]

    def get_top_n_with_scores(self, query: str, n: int = 5) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(idx, scores[idx]) for idx in ranked[:n]]

    def get_top_n_docs(self, query: str, n: int = 5) -> List[str]:
        return [self.docs[idx] for idx in self.get_top_n(query, n=n)]
