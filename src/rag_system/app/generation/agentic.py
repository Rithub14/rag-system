import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

TOOL_NAMES = [
    "summarize",
    "extract_facts",
    "compare",
    "generate_checklist",
    "draft_email",
    "find_tables",
    "list_definitions",
    "citations_by_section",
    "none",
]

DOC_ACTION_TOOLS = {
    "find_tables",
    "list_definitions",
    "citations_by_section",
}


def _safe_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _context_preview(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def select_tool(
    client: OpenAI,
    query: str,
    context_text: str,
    enable_doc_actions: bool = True,
) -> str:
    allowed = TOOL_NAMES.copy()
    if not enable_doc_actions:
        allowed = [name for name in allowed if name not in DOC_ACTION_TOOLS]

    prompt = {
        "role": "user",
        "content": (
            "Choose the best tool for the user query based on the context. "
            "Return JSON with keys: tool, reason. "
            f"Allowed tools: {', '.join(allowed)}.\n\n"
            f"Query: {query}\n\n"
            f"Context (preview):\n{_context_preview(context_text)}"
        ),
    }
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict tool router."},
            prompt,
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=120,
    )
    content = response.choices[0].message.content or ""
    data = _safe_json(content)
    tool = data.get("tool", "none")
    if tool not in allowed:
        return "none"
    return tool


def _run_llm_tool(
    client: OpenAI,
    tool: str,
    query: str,
    context_text: str,
    max_tokens: int = 400,
) -> str:
    system_map = {
        "summarize": "Summarize the context succinctly for the query. Keep citations.",
        "extract_facts": "Extract factual statements from the context with citations.",
        "compare": "Compare the key entities or options in the context. Use citations.",
        "generate_checklist": "Generate a checklist based on the context. Use citations.",
        "draft_email": "Draft a professional email using the context. Cite sources if relevant.",
    }
    system = system_map.get(tool, "You are a helpful assistant.")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nTask: {query}",
            },
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _find_tables(context_text: str) -> str:
    lines = context_text.splitlines()
    tables: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if "|" in line or "\t" in line:
            current.append(line)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    if not tables:
        return "No tables found in the provided context."
    blocks = ["\n".join(block) for block in tables]
    return "\n\n".join(blocks)


def _list_definitions(context_text: str) -> str:
    results = []
    for line in context_text.splitlines():
        match = re.match(r"^\s*([A-Za-z0-9][^:]{1,60}):\s+(.+)$", line)
        if match:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            results.append(f"- {term}: {definition}")
    if not results:
        return "No definition-style lines found in the provided context."
    return "\n".join(results)


def _citations_by_section(used_chunks: List[Dict[str, Any]]) -> str:
    if not used_chunks:
        return "No citations available."
    entries = []
    for chunk in used_chunks:
        source = chunk.get("source", "unknown")
        chunk_index = chunk.get("chunk_index", "0")
        snippet = (chunk.get("content", "") or "")[:160].replace("\n", " ")
        entries.append(f"[{source}#{chunk_index}] {snippet}")
    return "\n".join(entries)


def run_tool(
    client: OpenAI,
    tool: str,
    query: str,
    context_text: str,
    used_chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if tool in DOC_ACTION_TOOLS:
        if tool == "find_tables":
            return _find_tables(context_text)
        if tool == "list_definitions":
            return _list_definitions(context_text)
        if tool == "citations_by_section":
            return _citations_by_section(used_chunks or [])
    return _run_llm_tool(client, tool, query, context_text)


def plan_queries(
    client: OpenAI,
    query: str,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the query and propose up to 3 targeted retrieval queries. "
                    "Return JSON with keys: rewritten_query, entities, subqueries."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\nDoc ID: {doc_id or 'none'}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=200,
    )
    content = response.choices[0].message.content or ""
    data = _safe_json(content)
    rewritten = data.get("rewritten_query") or query
    subqueries = [
        q for q in (data.get("subqueries") or []) if isinstance(q, str)
    ][:3]
    queries = [rewritten] + [q for q in subqueries if q != rewritten]
    return {
        "rewritten_query": rewritten,
        "entities": data.get("entities") or [],
        "subqueries": subqueries,
        "queries": queries or [query],
    }


def generate_followups(
    client: OpenAI,
    query: str,
    answer: str,
    context_text: str,
) -> List[str]:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate 2-3 concise follow-up questions based on the answer "
                    "and missing context. Return JSON with key: follow_ups."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\nAnswer: {answer}\nContext:\n{_context_preview(context_text, 800)}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=120,
    )
    content = response.choices[0].message.content or ""
    data = _safe_json(content)
    follow_ups = [q for q in (data.get("follow_ups") or []) if isinstance(q, str)]
    return follow_ups[:3]
