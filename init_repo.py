from pathlib import Path

paths = [
    "app/api",
    "app/auth",
    "app/router",
    "app/retrieval",
    "app/rerank",
    "app/generation",
    "app/observability",
    "app/config",
    "data/raw_docs",
    "data/eval_set",
    "scripts",
    "tests",
]

files = [
    "app/main.py",
    "app/api/ingest.py",
    "app/api/query.py",
    "app/api/eval.py",
    "app/auth/jwt.py",
    "app/router/query_router.py",
    "app/retrieval/bm25.py",
    "app/retrieval/vector.py",
    "app/retrieval/hybrid.py",
    "app/retrieval/filters.py",
    "app/rerank/reranker.py",
    "app/generation/llm.py",
    "app/observability/langfuse.py",
    "app/observability/phoenix.py",
    "app/config/settings.py",
    "scripts/ingest_docs.py",
    "scripts/run_eval.py",
    "README.md",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml",
]

for p in paths:
    Path(p).mkdir(parents=True, exist_ok=True)

for f in files:
    Path(f).touch(exist_ok=True)

print("Repo structure created")
