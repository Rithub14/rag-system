# Enterprise RAG System

Enterprise RAG pipeline with user-scoped retrieval, reranking, tracing, metrics,
and offline evaluation.

## Quick start

1) Start Weaviate
```
docker compose up -d weaviate
```

2) Ingest docs (optional)
```
uv run -m scripts.ingest_docs
```

3) Run the API
```
PYTHONPATH=src uv run -m uvicorn enterprise_rag_system.main:app
```

4) Query
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
- `WEAVIATE_URL` (default `http://localhost:8080`)
- `WEAVIATE_GRPC_PORT` (default `50051`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `ARIZE_PHOENIX_API_KEY`, `ARIZE_PHOENIX_DEPLOYMENT`
- `APP_ENV`

See `.env.template`.

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

## Deployment (Docker + K8s + GH Actions)

Build images locally:
```
docker build -f Dockerfile.api -t rag-api:local .
docker build -f Dockerfile.streamlit -t rag-ui:local .
```

GitHub Actions builds and pushes to GHCR on `main`:
- `ghcr.io/<owner>/<repo>-api:latest`
- `ghcr.io/<owner>/<repo>-ui:latest`

Kubernetes manifests are in `k8s/`:
```
kubectl apply -f k8s/app-configmap.yaml
kubectl apply -f k8s/app-secret.example.yaml
kubectl apply -f k8s/redis-deployment.yaml -f k8s/redis-service.yaml
kubectl apply -f k8s/weaviate-deployment.yaml -f k8s/weaviate-service.yaml
kubectl apply -f k8s/api-deployment.yaml -f k8s/api-service.yaml
kubectl apply -f k8s/streamlit-deployment.yaml -f k8s/streamlit-service.yaml
kubectl apply -f k8s/monitoring/
```

Update image names in `k8s/*deployment.yaml` to match your GHCR repo.
