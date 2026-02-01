import os
import sys
from typing import Any, Dict

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "MCP server requires the 'mcp' package. "
        "Install with: uv pip install mcp"
    ) from exc


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_QUERY_URL = f"{API_BASE_URL}/api/query"

mcp = FastMCP("rag-system")


@mcp.tool()
def rag_query(
    query: str,
    k: int = 5,
    max_context_tokens: int = 1500,
    max_answer_tokens: int = 300,
    temperature: float = 0.2,
    rerank: bool = True,
    include_citations: bool = True,
    enable_tools: bool = True,
    enable_followups: bool = True,
    enable_planning: bool = False,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "k": k,
        "max_context_tokens": max_context_tokens,
        "max_answer_tokens": max_answer_tokens,
        "temperature": temperature,
        "rerank": rerank,
        "include_citations": include_citations,
        "enable_tools": enable_tools,
        "enable_followups": enable_followups,
        "enable_planning": enable_planning,
    }
    with httpx.Client(timeout=90) as client:
        resp = client.post(API_QUERY_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run()
