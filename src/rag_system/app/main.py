from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from uuid import uuid4

from .api.ingest import router as ingest_router
from .api.query import router as query_router
from .api.session import router as session_router
from .observability.langfuse import configure_langfuse_logging, get_langfuse
from .observability.logging import configure_json_logging, request_id_var
from .retrieval.vector import get_client
from .retrieval.vector_retriever import WeaviateRetriever

load_dotenv(".env")
configure_langfuse_logging()
configure_json_logging()

app = FastAPI(title="RAG System")

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response
    finally:
        request_id_var.reset(token)


@app.middleware("http")
async def ensure_browser_id(request: Request, call_next):
    browser_id = request.cookies.get("browser_id")
    if not browser_id:
        browser_id = str(uuid4())
    response = await call_next(request)
    if "browser_id" not in request.cookies:
        response.set_cookie(
            "browser_id",
            browser_id,
            httponly=True,
            samesite="lax",
        )
    return response


@app.on_event("startup")
def startup() -> None:
    # Defer Weaviate connection to first request so the API can start
    # even if Weaviate is sleeping or temporarily unavailable.
    app.state.weaviate_client = None
    app.state.vector_retriever = None
    app.state.langfuse = get_langfuse(raise_if_configured=True)


@app.on_event("shutdown")
def shutdown() -> None:
    client = getattr(app.state, "weaviate_client", None)
    if client is not None:
        client.close()
    langfuse = getattr(app.state, "langfuse", None)
    if langfuse is not None and hasattr(langfuse, "flush"):
        langfuse.flush()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(query_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(session_router, prefix="/api")
