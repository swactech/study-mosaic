"""
Lightweight wrapper around a local vector store.

Uses a custom Gemini embedding function with manual retries for transient errors.
"""

import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import chromadb  # type: ignore
import google.generativeai as genai  # type: ignore
from google.genai import types as genai_types
from app.config import settings


class GeminiEmbeddingFunction:
    """Minimal embedding function Chroma can call with retries."""

    def __init__(self, model_name: str, api_key_env: str):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing API key env var {api_key_env}. Set it in your shell or .env."
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.retry_config = genai_types.HttpRetryOptions(
            attempts=5,
            exp_base=7,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A003
        embeddings: List[List[float]] = []
        for text in input:
            attempt = 0
            last_exc = None
            while attempt < self.retry_config.attempts:
                try:
                    response = genai.embed_content(
                        model=self.model_name,
                        content=text,
                    )
                    embeddings.append(response["embedding"])
                    break
                except Exception as exc:  # pragma: no cover - network/model issues
                    last_exc = exc
                    attempt += 1
                    if attempt >= self.retry_config.attempts:
                        raise
                    delay = self.retry_config.initial_delay * (
                        self.retry_config.exp_base ** (attempt - 1)
                    )
                    # Add small jitter to avoid thundering herd.
                    time.sleep(delay + random.uniform(0, 0.5))
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Chroma compatibility: embed a list of documents."""
        return self(texts)

    def embed_query(self, input: str) -> List[float]:
        """Chroma compatibility: embed a single query string."""
        return self([input])[0]

    def name(self) -> str:
        """Expose a stable name to satisfy Chroma's embedding function checks."""
        return "gemini-embedding-function"


class LocalVectorStore:
    """Persist embeddings locally per session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.store_path = Path(settings.vector_store_dir)
        self.store_path.mkdir(parents=True, exist_ok=True)

        embedding_fn = GeminiEmbeddingFunction(
            model_name=settings.embedding_model,
            api_key_env=settings.api_key_env,
        )

        self.client = chromadb.PersistentClient(path=str(self.store_path))
        self.collection = self.client.get_or_create_collection(
            name=session_id,
            embedding_function=embedding_fn,
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]] = None,
        documents: List[str] = None,
        metadatas: List[dict] = None,
    ):
        """
        Add documents; embeddings are optional because Chroma can compute them via embedding_function.
        """
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query: str, top_k: int) -> List[Tuple[str, str, dict, float]]:
        """
        Search the store and return (id, document, metadata, score).
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        hits = []
        for idx, doc_id in enumerate(results.get("ids", [[]])[0]):
            doc = results["documents"][0][idx]
            meta = results["metadatas"][0][idx]
            score = results["distances"][0][idx]
            hits.append((doc_id, doc, meta, score))
        return hits

    def count(self) -> int:
        """Return how many chunks are stored for the session."""
        return self.collection.count()

    def list_metadatas(self) -> List[dict]:
        """Return stored chunk metadata for the session."""
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        # Some Chroma versions return a nested list, flatten if needed.
        if metadatas and isinstance(metadatas[0], list):
            flat: List[dict] = []
            for meta in metadatas:
                if isinstance(meta, list):
                    flat.extend(m for m in meta if m)
                elif meta:
                    flat.append(meta)
            metadatas = flat
        return [m for m in metadatas if m]
