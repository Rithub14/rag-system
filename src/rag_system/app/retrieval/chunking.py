from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .schema import DocumentChunk


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_text(text)

    return [
        DocumentChunk(
            content=chunk,
            metadata={**metadata, "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]
