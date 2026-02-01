from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of RAG queries",
    ["endpoint"],
)

ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Total number of errors by endpoint",
    ["endpoint"],
)

LATENCY_SECONDS = Histogram(
    "rag_latency_seconds",
    "Latency by stage in seconds",
    ["stage"],
)

TOKENS_TOTAL = Counter(
    "rag_tokens_total",
    "Total tokens used by type",
    ["type"],
)

QUERY_LENGTH = Histogram(
    "rag_query_length_chars",
    "Query length in characters",
)

CONTEXT_LENGTH = Histogram(
    "rag_context_length_chars",
    "Context length in characters",
)

RETRIEVED_COUNT = Histogram(
    "rag_retrieved_count",
    "Number of retrieved chunks",
)

RERANKED_COUNT = Histogram(
    "rag_reranked_count",
    "Number of reranked chunks",
)

USED_COUNT = Histogram(
    "rag_used_count",
    "Number of chunks used in context",
)
