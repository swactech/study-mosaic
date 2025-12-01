"""
PDF ingestion: extract text, chunk, and persist in the local vector store.
"""

from pathlib import Path
from typing import Dict, List

import pdfplumber  # type: ignore

from app.config import settings
from app.tools.vector_store import LocalVectorStore


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Simple character-based chunker that preserves offsets."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "char_start": start, "char_end": end})
        if end == len(text):
            break
        start = end - overlap
    return chunks


def ingest_pdfs(session_id: str, pdf_paths: List[Path]) -> List[Dict]:
    """
    Ingest 1–5 PDFs, chunk, and store them in the local vector store.

    Returns a list of chunk metadata for downstream citation mapping.
    """
    if len(pdf_paths) == 0:
        raise ValueError("No PDF paths provided.")
    if len(pdf_paths) > settings.max_pdfs_per_session:
        raise ValueError(
            f"Too many PDFs. Max allowed per session: {settings.max_pdfs_per_session}"
        )

    store = LocalVectorStore(session_id=session_id)
    all_chunks: List[Dict] = []

    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue

                chunks = _chunk_text(
                    page_text, settings.chunk_size, settings.chunk_overlap
                )
                for local_idx, chunk in enumerate(chunks):
                    chunk_id = f"{pdf_path.stem}-p{page_index}-c{local_idx}"
                    metadata = {
                        "chunk_id": chunk_id,
                        "page": page_index,
                        "pdf": pdf_path.name,
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    }
                    store.add(
                        ids=[chunk_id],
                        documents=[chunk["text"]],
                        metadatas=[metadata],
                    )
                    all_chunks.append(
                        {
                            "id": chunk_id,
                            "text": chunk["text"],
                            "page": page_index,
                            "pdf": pdf_path.name,
                            "char_start": chunk["char_start"],
                            "char_end": chunk["char_end"],
                        }
                    )
    return all_chunks
