from typing import Any, Dict, List, Optional

import weaviate

from .embeddings import embed_texts


class WeaviateRetriever:
    def __init__(self, client: weaviate.WeaviateClient, class_name: str = "DocumentChunk") -> None:
        self.client = client
        self.class_name = class_name
        self.collection = client.collections.get(class_name)

    def query(
        self,
        query_text: str,
        k: int = 10,
        *,
        filters: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        vector = embed_texts([query_text])[0]
        response = self.collection.query.near_vector(
            near_vector=vector,
            limit=k,
            filters=filters,
            return_properties=[
                "content",
                "user_id",
                "doc_id",
                "source",
                "chunk_index",
            ],
        )
        return [obj.properties for obj in response.objects]
