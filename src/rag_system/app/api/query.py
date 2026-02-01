import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from ..generation.agentic import (
    generate_followups,
    plan_queries,
    run_tool,
    select_tool,
)
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.faiss_store import get_store
from ..retrieval.embeddings import embed_texts
from ..retrieval.reranker import rerank_with_scores
from ..response.context_builder import build_context
from ..observability.metrics import (
    CONTEXT_LENGTH,
    ERRORS_TOTAL,
    LATENCY_SECONDS,
    QUERY_LENGTH,
    REQUESTS_TOTAL,
    RERANKED_COUNT,
    RETRIEVED_COUNT,
    TOKENS_TOTAL,
    USED_COUNT,
)
from ..observability.logging import trace_id_var
from ..observability.ratelimit import rate_limiter

router = APIRouter()
logger = logging.getLogger("rag")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    doc_id: Optional[str] = None
    max_context_tokens: int = Field(1500, ge=200, le=6000)
    max_answer_tokens: int = Field(300, ge=50, le=1000)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    rerank: bool = True
    include_citations: bool = True
    enable_tools: Optional[bool] = None
    enable_followups: Optional[bool] = None
    enable_planning: Optional[bool] = None


class QueryResult(BaseModel):
    content: str
    user_id: Optional[str] = None
    doc_id: Optional[str] = None
    source: Optional[str] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    context: str
    citations: Dict[str, List[Dict[str, Optional[str]]]]
    results: List[QueryResult]
    tool_used: Optional[str] = None
    tool_output: Optional[str] = None
    follow_ups: List[str] = []
    plan: Optional[Dict[str, Any]] = None


@router.post("/query", response_model=QueryResponse)
def query_docs(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    REQUESTS_TOTAL.labels(endpoint="/api/query").inc()
    browser_id = request.cookies.get("browser_id")
    session_id = request.headers.get("x-session-id")
    client_ip = request.headers.get("x-forwarded-for", request.client.host)
    limiter_key = browser_id or session_id or client_ip
    rate_limiter.check("query", limiter_key, limit=10, window_seconds=3600)
    user_context = {
        "user_id": browser_id or session_id or client_ip,
    }
    langfuse = getattr(request.app.state, "langfuse", None)
    trace = None
    trace_token = None
    if langfuse is not None:
        trace = langfuse.trace(
            name="rag_query",
            user_id=user_context["user_id"],
            metadata={
                "doc_id": payload.doc_id,
            },
            input={
                "query": payload.query,
                "k": payload.k,
            },
        )
        trace_id = getattr(trace, "id", None)
        if trace_id:
            trace_token = trace_id_var.set(trace_id)

    def env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def chunk_id(chunk: Dict) -> str:
        return f"{chunk.get('source', 'unknown')}#{chunk.get('chunk_index', '0')}"

    # Query reception span
    if trace is not None:
        span = trace.span(
            name="query_reception",
            input={
                "query": payload.query,
                "user_id": user_context["user_id"],
                "doc_id": payload.doc_id,
            },
        )
        span.end(metadata={"latency_ms": 0})

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIError(
                "OPENAI_API_KEY is not set. Export it or set it in your .env file."
            )
        client = OpenAI(api_key=api_key)
        enable_tools = (
            payload.enable_tools
            if payload.enable_tools is not None
            else env_flag("ENABLE_TOOL_ROUTER", True)
        )
        enable_followups = (
            payload.enable_followups
            if payload.enable_followups is not None
            else env_flag("ENABLE_FOLLOWUPS", True)
        )
        enable_planning = (
            payload.enable_planning
            if payload.enable_planning is not None
            else env_flag("ENABLE_PLANNING", False)
        )
        enable_doc_actions = env_flag("ENABLE_DOC_ACTIONS", True)

        try:
            store = get_store()
        except Exception as exc:
            ERRORS_TOTAL.labels(endpoint="/api/query").inc()
            raise HTTPException(
                status_code=503, detail="Vector store is unavailable."
            ) from exc

        plan = None
        planned_queries = [payload.query]
        if enable_planning:
            planning_span = trace.span(name="planning") if trace else None
            planning_start = time.perf_counter()
            plan = plan_queries(client, payload.query, payload.doc_id)
            planned_queries = plan.get("queries") or planned_queries
            planning_latency = (time.perf_counter() - planning_start) * 1000
            LATENCY_SECONDS.labels(stage="planning").observe(
                planning_latency / 1000
            )
            logger.info(
                "agentic_planning",
                extra={
                    "user_id": user_context["user_id"],
                    "queries": planned_queries,
                    "doc_id": payload.doc_id,
                },
            )
            if planning_span is not None:
                planning_span.end(
                    metadata={"latency_ms": round(planning_latency, 2)},
                    output=plan,
                )

        # Dense retrieval
        dense_span = (
            trace.span(name="dense_retrieval", input={"k": payload.k})
            if trace
            else None
        )
        dense_start = time.perf_counter()
        try:
            query_vecs = embed_texts(planned_queries)
        except OpenAIError as exc:
            ERRORS_TOTAL.labels(endpoint="/api/query").inc()
            raise HTTPException(
                status_code=502,
                detail="Embedding provider error. Check OPENAI_API_KEY.",
            ) from exc

        try:
            dense_results = []
            seen_ids = set()
            for vec in query_vecs:
                results = store.search(
                    query_vector=vec,
                    k=payload.k,
                    user_id=user_context["user_id"],
                    doc_id=payload.doc_id,
                )
                for doc in results:
                    doc_key = chunk_id(doc)
                    if doc_key in seen_ids:
                        continue
                    seen_ids.add(doc_key)
                    dense_results.append(doc)
        except Exception as exc:
            ERRORS_TOTAL.labels(endpoint="/api/query").inc()
            raise HTTPException(
                status_code=503, detail="Vector store is unavailable."
            ) from exc
        dense_latency = (time.perf_counter() - dense_start) * 1000
        LATENCY_SECONDS.labels(stage="dense_retrieval").observe(dense_latency / 1000)
        dense_ids = [chunk_id(doc) for doc in dense_results]
        RETRIEVED_COUNT.observe(len(dense_results))
        if dense_span is not None:
            dense_span.end(
                metadata={"latency_ms": round(dense_latency, 2)},
                output={"chunk_ids": dense_ids},
            )

        filtered_results = dense_results

        # BM25 / sparse retrieval (over filtered dense results for now)
        bm25_span = trace.span(name="bm25_retrieval") if trace else None
        bm25_start = time.perf_counter()
        if filtered_results:
            bm25 = BM25Retriever([doc.get("content", "") for doc in filtered_results])
            bm25_scores = bm25.get_top_n_with_scores(
                payload.query, n=len(filtered_results)
            )
            bm25_ranked = [filtered_results[i] for i, _ in bm25_scores]
            bm25_ranked_ids = [chunk_id(doc) for doc in bm25_ranked]
            bm25_score_pairs = [
                {"chunk_id": chunk_id(filtered_results[i]), "score": float(score)}
                for i, score in bm25_scores
            ]
        else:
            bm25_ranked = []
            bm25_ranked_ids = []
            bm25_score_pairs = []
        bm25_latency = (time.perf_counter() - bm25_start) * 1000
        LATENCY_SECONDS.labels(stage="bm25_retrieval").observe(bm25_latency / 1000)
        if bm25_span is not None:
            bm25_span.end(
                metadata={"latency_ms": round(bm25_latency, 2)},
                output={"chunk_ids": bm25_ranked_ids, "scores": bm25_score_pairs},
            )

        # Reranking
        rerank_span = trace.span(name="reranking") if trace else None
        rerank_start = time.perf_counter()
        if payload.rerank:
            reranked_with_scores = rerank_with_scores(payload.query, filtered_results)
            reranked = [doc for doc, _ in reranked_with_scores]
            rerank_scores = [
                {"chunk_id": chunk_id(doc), "score": float(score)}
                for doc, score in reranked_with_scores
            ]
        else:
            reranked = filtered_results
            rerank_scores = []
        rerank_latency = (time.perf_counter() - rerank_start) * 1000
        LATENCY_SECONDS.labels(stage="reranking").observe(rerank_latency / 1000)
        RERANKED_COUNT.observe(len(reranked))
        if rerank_span is not None:
            rerank_span.end(
                metadata={"latency_ms": round(rerank_latency, 2)},
                output={"scores": rerank_scores},
            )

        # Context building
        context_span = trace.span(name="context_building") if trace else None
        context_start = time.perf_counter()
        context_text, used_chunks = build_context(
            payload.query, reranked, max_tokens=payload.max_context_tokens
        )
        context_latency = (time.perf_counter() - context_start) * 1000
        LATENCY_SECONDS.labels(stage="context_building").observe(
            context_latency / 1000
        )
        CONTEXT_LENGTH.observe(len(context_text))
        USED_COUNT.observe(len(used_chunks))
        if context_span is not None:
            context_span.end(
                metadata={"latency_ms": round(context_latency, 2)},
                output={"chunk_ids": [chunk_id(doc) for doc in used_chunks]},
            )

        tool_used = None
        tool_output = None
        if enable_tools and context_text:
            tool_used = select_tool(
                client,
                payload.query,
                context_text,
                enable_doc_actions=enable_doc_actions,
            )
            if tool_used == "none":
                tool_used = None
            if tool_used:
                tool_output = run_tool(
                    client,
                    tool_used,
                    payload.query,
                    context_text,
                    used_chunks,
                )
                logger.info(
                    "agentic_tool_used",
                    extra={
                        "user_id": user_context["user_id"],
                        "tool": tool_used,
                        "doc_id": payload.doc_id,
                        "output_chars": len(tool_output or ""),
                    },
                )

        gen_span = trace.span(name="generation") if trace else None
        gen_start = time.perf_counter()
        tool_block = (
            f"\n\nTool output ({tool_used}):\n{tool_output}\n"
            if tool_used and tool_output
            else ""
        )
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an enterprise RAG assistant."},
                {
                    "role": "user",
                    "content": f"{context_text}{tool_block}\n\nQuestion: {payload.query}",
                },
            ],
            max_tokens=payload.max_answer_tokens,
            temperature=payload.temperature,
        )
        answer = completion.choices[0].message.content or ""
        gen_latency = (time.perf_counter() - gen_start) * 1000
        LATENCY_SECONDS.labels(stage="generation").observe(gen_latency / 1000)
        if gen_span is not None:
            usage = getattr(completion, "usage", None)
            gen_span.end(
                metadata={
                    "latency_ms": round(gen_latency, 2),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None)
                    if usage
                    else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None)
                    if usage
                    else None,
                    "total_tokens": getattr(usage, "total_tokens", None)
                    if usage
                    else None,
                },
            )
        usage = getattr(completion, "usage", None)
        if usage:
            if getattr(usage, "prompt_tokens", None) is not None:
                TOKENS_TOTAL.labels(type="prompt").inc(int(usage.prompt_tokens))
            if getattr(usage, "completion_tokens", None) is not None:
                TOKENS_TOTAL.labels(type="completion").inc(
                    int(usage.completion_tokens)
                )
            if getattr(usage, "total_tokens", None) is not None:
                TOKENS_TOTAL.labels(type="total").inc(int(usage.total_tokens))
        QUERY_LENGTH.observe(len(payload.query))
        logger.info(
            "query_completed",
            extra={
                "user_id": user_context["user_id"],
                "query_length": len(payload.query),
                "retrieved": len(dense_results),
                "reranked": len(reranked),
                "used": len(used_chunks),
                "prompt_tokens": getattr(usage, "prompt_tokens", None)
                if usage
                else None,
                "completion_tokens": getattr(usage, "completion_tokens", None)
                if usage
                else None,
                "total_tokens": getattr(usage, "total_tokens", None)
                if usage
                else None,
            },
        )
        follow_ups: List[str] = []
        if enable_followups:
            followups_span = trace.span(name="followups") if trace else None
            followups_start = time.perf_counter()
            follow_ups = generate_followups(
                client,
                payload.query,
                answer,
                context_text,
            )
            followups_latency = (time.perf_counter() - followups_start) * 1000
            LATENCY_SECONDS.labels(stage="followups").observe(
                followups_latency / 1000
            )
            logger.info(
                "agentic_followups",
                extra={
                    "user_id": user_context["user_id"],
                    "doc_id": payload.doc_id,
                    "count": len(follow_ups),
                },
            )
            if followups_span is not None:
                followups_span.end(
                    metadata={"latency_ms": round(followups_latency, 2)},
                    output={"follow_ups": follow_ups},
                )
    except Exception:
        ERRORS_TOTAL.labels(endpoint="/api/query").inc()
        logger.exception("query_failed")
        raise
    finally:
        if trace_token is not None:
            trace_id_var.reset(trace_token)

    if payload.include_citations:
        citations_used = [
            {
                "source": chunk.get("source"),
                "chunk_index": str(chunk.get("chunk_index"))
                if chunk.get("chunk_index") is not None
                else None,
            }
            for chunk in used_chunks
        ]
        citations_related = [
            {
                "source": chunk.get("source"),
                "chunk_index": str(chunk.get("chunk_index"))
                if chunk.get("chunk_index") is not None
                else None,
            }
            for chunk in reranked
            if chunk not in used_chunks
        ]
    else:
        citations_used = []
        citations_related = []
    return QueryResponse(
        query=payload.query,
        answer=answer,
        context=context_text,
        citations={"used": citations_used, "related": citations_related},
        results=[QueryResult(**result) for result in filtered_results],
        tool_used=tool_used,
        tool_output=tool_output,
        follow_ups=follow_ups,
        plan=plan,
    )
