"""
Retrieval tool: semantic search over the local vector store.
"""

from typing import List, Dict

from app.config import settings
from app.tools.vector_store import LocalVectorStore


def retrieve_chunks(session_id: str, query: str, top_k: int = None) -> List[Dict]:
    """Return ranked chunks with metadata for grounding."""
    store = LocalVectorStore(session_id=session_id)
    k = top_k or settings.top_k
    hits = store.query(query=query, top_k=k)
    results: List[Dict] = []
    for chunk_id, document, meta, score in hits:
        results.append(
            {
                "id": chunk_id,
                "text": document,
                "page": meta.get("page"),
                "pdf": meta.get("pdf"),
                "char_start": meta.get("char_start"),
                "char_end": meta.get("char_end"),
                "score": score,
            }
        )
    return results
