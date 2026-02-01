import os
import sqlite3
import threading
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np


class FaissStore:
    def __init__(self, db_path: str, index_path: str) -> None:
        self.db_path = db_path
        self.index_path = index_path
        self._lock = threading.Lock()
        self._ensure_dirs()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._index = self._load_or_build_index()

    def _ensure_dirs(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        idx_dir = os.path.dirname(self.index_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        if idx_dir:
            os.makedirs(idx_dir, exist_ok=True)

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                doc_id TEXT,
                source TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON chunks(user_id, doc_id)"
        )
        self._conn.commit()

    def _load_or_build_index(self) -> Optional[faiss.Index]:
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception:
                pass
        rows = self._conn.execute("SELECT id, embedding FROM chunks").fetchall()
        if not rows:
            return None
        ids = np.array([row["id"] for row in rows], dtype="int64")
        vectors = np.vstack(
            [np.frombuffer(row["embedding"], dtype="float32") for row in rows]
        )
        if vectors.size == 0:
            return None
        faiss.normalize_L2(vectors)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, ids)
        faiss.write_index(index, self.index_path)
        return index

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings:
            return 0
        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        ids: List[int] = []
        with self._conn:
            for chunk, vec in zip(chunks, vectors):
                cursor = self._conn.execute(
                    """
                    INSERT INTO chunks (user_id, doc_id, source, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.get("user_id"),
                        chunk.get("doc_id"),
                        chunk.get("source"),
                        chunk.get("chunk_index"),
                        chunk.get("content"),
                        vec.tobytes(),
                    ),
                )
                ids.append(int(cursor.lastrowid))
        if not ids:
            return 0
        id_array = np.array(ids, dtype="int64")
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexIDMap2(
                    faiss.IndexFlatIP(vectors.shape[1])
                )
            if self._index.d != vectors.shape[1]:
                # Rebuild index if dimensions mismatch
                self._index = self._load_or_build_index()
            if self._index is None:
                self._index = faiss.IndexIDMap2(
                    faiss.IndexFlatIP(vectors.shape[1])
                )
            self._index.add_with_ids(vectors, id_array)
            faiss.write_index(self._index, self.index_path)
        return len(ids)

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        user_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        query = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        search_k = min(max(k * 5, k), int(self._index.ntotal))
        with self._lock:
            scores, ids = self._index.search(query, search_k)
        id_list = [int(i) for i in ids[0] if i != -1]
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, doc_id, source, chunk_index, content
            FROM chunks WHERE id IN ({placeholders})
            """,
            id_list,
        ).fetchall()
        row_map = {row["id"]: row for row in rows}
        results: List[Dict] = []
        for idx in id_list:
            row = row_map.get(idx)
            if row is None:
                continue
            if user_id and row["user_id"] != user_id:
                continue
            if doc_id and row["doc_id"] != doc_id:
                continue
            results.append(
                {
                    "content": row["content"],
                    "user_id": row["user_id"],
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                }
            )
            if len(results) >= k:
                break
        return results

    def close(self) -> None:
        self._conn.close()


_store: Optional[FaissStore] = None


def get_store() -> FaissStore:
    global _store
    if _store is None:
        db_path = os.getenv("RAG_DB_PATH", "data/rag.db")
        index_path = os.getenv("RAG_INDEX_PATH", "data/faiss.index")
        _store = FaissStore(db_path, index_path)
    return _store
