from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from src.enterprise_rag_system.app.retrieval.reranker import rerank_with_scores
from src.enterprise_rag_system.app.retrieval.vector import get_client
from src.enterprise_rag_system.app.retrieval.vector_retriever import WeaviateRetriever
from src.enterprise_rag_system.app.response.context_builder import build_context
from weaviate.classes.query import Filter


@dataclass
class EvalCase:
    query: str
    user_id: str
    doc_id: str
    expected_sources: List[str]


def load_cases(path: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                query=payload["query"],
                user_id=payload["user_id"],
                doc_id=payload["doc_id"],
                expected_sources=payload.get("expected_sources", []),
            )
        )
    return cases


def chunk_id(chunk: Dict) -> str:
    return f"{chunk.get('source', 'unknown')}#{chunk.get('chunk_index', '0')}"


def run_case(
    client: WeaviateRetriever,
    case: EvalCase,
    k: int = 5,
) -> Tuple[Dict, Dict]:
    filters = Filter.by_property("user_id").equal(case.user_id) & Filter.by_property(
        "doc_id"
    ).equal(case.doc_id)
    dense = client.query(case.query, k=k, filters=filters)
    filtered = dense
    reranked_with_scores = rerank_with_scores(case.query, filtered)
    reranked = [doc for doc, _ in reranked_with_scores]
    context_text, used_chunks = build_context(case.query, reranked)

    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIError("OPENAI_API_KEY is not set.")
    llm = OpenAI(api_key=api_key)
    completion = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an enterprise RAG assistant."},
            {"role": "user", "content": f"{context_text}\n\nQuestion: {case.query}"},
        ],
        max_tokens=300,
    )
    answer = completion.choices[0].message.content or ""

    used_ids = [chunk_id(c) for c in used_chunks]
    expected_set = set(case.expected_sources)
    hit_count = len(expected_set.intersection(used_ids))
    recall = hit_count / max(1, len(expected_set))

    result = {
        "query": case.query,
        "answer": answer,
        "citations_used": used_ids,
        "expected_sources": case.expected_sources,
        "retrieval_recall": recall,
    }
    metrics = {
        "recall": recall,
    }
    return result, metrics


def main() -> None:
    load_dotenv(".env")
    cases = load_cases(Path("data/eval/queries.jsonl"))
    if not cases:
        print("No eval cases found.")
        return

    weaviate_client = get_client()
    retriever = WeaviateRetriever(weaviate_client)
    results = []
    recalls = []

    try:
        for case in cases:
            result, metrics = run_case(retriever, case, k=5)
            results.append(result)
            recalls.append(metrics["recall"])
    finally:
        weaviate_client.close()

    avg_recall = sum(recalls) / max(1, len(recalls))
    report = {
        "num_cases": len(cases),
        "avg_recall": avg_recall,
        "cases": results,
    }

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "eval_report.md").write_text(
        f"# Offline Eval Report\n\n"
        f"- cases: {len(cases)}\n"
        f"- avg_recall: {avg_recall:.3f}\n"
    )
    print("Wrote reports/eval_report.json and reports/eval_report.md")


if __name__ == "__main__":
    main()
