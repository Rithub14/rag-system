from dotenv import load_dotenv

load_dotenv() 

from pathlib import Path

from src.enterprise_rag_system.app.retrieval.chunking import chunk_text
from src.enterprise_rag_system.app.retrieval.embeddings import embed_texts
from src.enterprise_rag_system.app.retrieval.vector import get_client, ensure_schema, store_chunks


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def main() -> None:
    docs_path = Path("data/raw_docs")

    client = get_client()
    try:
        ensure_schema(client)

        for file in docs_path.glob("*.txt"):
            text = load_text(file)

        metadata = {
            "source": file.name,
            "user_id": "seed",
            "doc_id": file.stem,
        }

            chunks = chunk_text(text, metadata)
            embeddings = embed_texts([c.content for c in chunks])

            store_chunks(client, chunks, embeddings)

            print(f"Ingested {file.name} ({len(chunks)} chunks)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
