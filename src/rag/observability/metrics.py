from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Counters
QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["status"],
)

DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested",
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Total cache hits",
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Total cache misses",
)

# Histograms
QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "RAG query duration in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RETRIEVAL_SCORES = Histogram(
    "rag_retrieval_scores",
    "Distribution of retrieval similarity scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def setup_metrics(app):
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/api/v1/health"],
    ).instrument(app).expose(app, endpoint="/metrics")
