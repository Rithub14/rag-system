from typing import Dict, List, Tuple


def build_context(
    query: str,
    reranked: List[Dict],
    max_tokens: int = 1500,
) -> Tuple[str, List[Dict]]:
    content_accum = ""
    used_chunks: List[Dict] = []
    for chunk in reranked:
        source = chunk.get("source", "unknown")
        chunk_index = chunk.get("chunk_index", "0")
        text = f"[{source}#{chunk_index}] {chunk.get('content', '')}\n"
        if len(content_accum) + len(text) > max_tokens:
            break
        content_accum += text
        used_chunks.append(chunk)
    return content_accum, used_chunks
