# RAG System

RAG pipeline with SQLite + FAISS storage, user-scoped retrieval, reranking,
tracing, metrics, and offline evaluation.

## Quick start

1) Ingest docs (optional)
```
uv run -m scripts.ingest_docs
```

2) Run the API
```
PYTHONPATH=src uv run -m uvicorn rag_system.main:app
```

3) Query
```
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user1" \
  -d '{"query":"example text","k":5}'
```

## Streamlit UI

```
uv run -m streamlit run streamlit_app.py
```

Live demo: [rag-ui-0d2k.onrender.com](https://rag-ui-0d2k.onrender.com)

## PDF ingestion

```
curl -X POST "http://127.0.0.1:8000/api/ingest/file" \
  -H "X-Session-Id: user1" \
  -F "file=@/path/to/file.pdf"
```

OCR fallback (for scanned PDFs) requires system deps:
- macOS: `brew install tesseract poppler`

Upload limits:
- `MAX_UPLOAD_MB` (default 10 MB)

## Environment

Required:
- `OPENAI_API_KEY`

Optional:
- `RAG_DB_PATH` (default `data/rag.db`)
- `RAG_INDEX_PATH` (default `data/faiss.index`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `APP_ENV`

See `.env.template`.

## Agentic features

- Tool router in the answer stage (summarize, extract facts, compare, checklist, draft email).
- Document-aware tools (find tables, list definitions, citations by section).
- Follow-up question generator (2–3 suggestions after each answer).
- Optional query planning + retrieval refinement (toggle).

Toggles:
- `ENABLE_TOOL_ROUTER`, `ENABLE_DOC_ACTIONS`
- `ENABLE_FOLLOWUPS`
- `ENABLE_PLANNING` (off by default)

## Tracing (Langfuse)

Traces include spans for:
query reception, dense retrieval, BM25, reranking, context building, generation.
Each span logs timing and metadata (chunk IDs, rerank scores, token counts).

## Metrics (Prometheus)

Expose metrics at:
```
http://127.0.0.1:8000/metrics
```

Key metrics:
- `rag_requests_total`, `rag_errors_total`
- `rag_latency_seconds` (stages)
- `rag_tokens_total`
- `rag_query_length_chars`, `rag_context_length_chars`
- `rag_retrieved_count`, `rag_reranked_count`, `rag_used_count`

Rate limits:
- 10 queries per browser per hour (cookie-based; IP fallback)
- 1 PDF upload per browser per hour (cookie-based; IP fallback)

Redis (recommended for persistent rate limits):
- Start: `docker compose up -d redis`
- Set `REDIS_URL=redis://localhost:6379/0`

### Local Prometheus + Grafana

```
docker compose up -d prometheus grafana
```

Prometheus: `http://localhost:9090`  
Grafana: `http://localhost:3000` (admin/admin)

## Offline evaluation

Edit `data/eval/queries.jsonl`, then run:
```
PYTHONPATH=src uv run -m scripts.eval_offline
```

Reports:
- `reports/eval_report.json`
- `reports/eval_report.md`

## Docker build

Build images locally:
```
docker build -f Dockerfile.api -t rag-api:local .
docker build -f Dockerfile.streamlit -t rag-ui:local .
```
