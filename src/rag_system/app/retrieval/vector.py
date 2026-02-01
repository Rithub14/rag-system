import os
from typing import List
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from .schema import DocumentChunk


def get_client() -> weaviate.WeaviateClient:
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    parsed = urlparse(weaviate_url)
    http_host = parsed.hostname or "localhost"
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    http_secure = parsed.scheme == "https"
    grpc_host = os.getenv("WEAVIATE_GRPC_HOST", http_host)
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    grpc_secure = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"

    if grpc_host == http_host and grpc_port == http_port:
        grpc_port = 50051 if http_port != 50051 else 50052

    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
    )


def ensure_schema(client: weaviate.WeaviateClient) -> None:
    class_name = "DocumentChunk"

    existing = client.collections.list_all(simple=True)
    if isinstance(existing, dict):
        collection_names = set(existing.keys())
    else:
        collection_names = set(existing)

    if class_name not in collection_names:
        client.collections.create(
            name=class_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="user_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )
        return

    collection = client.collections.get(class_name)
    existing_props = {prop.name for prop in collection.config.get().properties or []}
    required = {
        "content": DataType.TEXT,
        "user_id": DataType.TEXT,
        "doc_id": DataType.TEXT,
        "source": DataType.TEXT,
        "chunk_index": DataType.INT,
    }
    for name, dtype in required.items():
        if name not in existing_props:
            collection.config.add_property(Property(name=name, data_type=dtype))


def store_chunks(
    client: weaviate.WeaviateClient,
    chunks: List[DocumentChunk],
    embeddings: List[list[float]],
) -> None:
    collection = client.collections.get("DocumentChunk")
    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk, vector in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "content": chunk.content,
                    **chunk.metadata,
                },
                vector=vector,
            )
