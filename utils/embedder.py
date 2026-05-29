"""
utils/embedder.py — Embedding management for LawAI India.

Wraps sentence-transformers/all-MiniLM-L6-v2 with ChromaDB persistence.
Avoids recomputation by tracking ingested sources.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lawai_india"


class EmbeddingManager:
    """
    Manages embeddings using HuggingFace sentence-transformers and ChromaDB.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for embeddings.
    chroma_dir : str
        Path for persistent ChromaDB storage.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chroma_dir: str = "chroma_db",
    ) -> None:
        self.model_name = model_name
        self.chroma_dir = chroma_dir

        logger.info("Loading embedding model: %s", model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Persistent ChromaDB client
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self._client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        logger.info("ChromaDB initialised at %s", chroma_dir)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Embed and store documents, skipping already-ingested chunks.

        Parameters
        ----------
        documents : list[Document]
            LangChain documents with page_content and metadata.
        batch_size : int
            Number of documents to embed per batch.

        Returns
        -------
        int
            Number of new chunks actually added.
        """
        if not documents:
            return 0

        # Filter duplicates based on content hash
        existing_ids = self._get_existing_ids()
        new_docs: List[Document] = []
        new_ids: List[str] = []

        for doc in documents:
            doc_id = self._chunk_id(doc)
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)

        if not new_docs:
            logger.info("All %d documents already indexed — skipping.", len(documents))
            return 0

        logger.info("Adding %d new chunks (skipping %d duplicates)…",
                    len(new_docs), len(documents) - len(new_docs))

        # Batch ingestion
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            batch_ids = new_ids[i : i + batch_size]
            self.vectorstore.add_documents(documents=batch, ids=batch_ids)
            logger.debug("Batch %d/%d ingested.", i // batch_size + 1,
                         (len(new_docs) - 1) // batch_size + 1)

        logger.info("Ingestion complete: %d chunks added.", len(new_docs))
        return len(new_docs)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        try:
            collection = self._client.get_collection(COLLECTION_NAME)
            count = collection.count()

            # Unique acts and sources
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas") or []

            acts = sorted({
                m.get("act_name", "Unknown")
                for m in metadatas
                if m.get("act_name")
            })
            sources = {
                m.get("source", "")
                for m in metadatas
                if m.get("source")
            }

            return {
                "num_chunks": count,
                "num_embeddings": count,
                "num_documents": len(sources),
                "acts": acts,
            }
        except Exception as exc:
            logger.warning("Could not fetch stats: %s", exc)
            return {
                "num_chunks": 0,
                "num_embeddings": 0,
                "num_documents": 0,
                "acts": [],
            }

    def get_vectorstore(self) -> Chroma:
        """Return the underlying LangChain Chroma vectorstore."""
        return self.vectorstore

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _chunk_id(self, doc: Document) -> str:
        """Generate a deterministic ID from content + source + page."""
        payload = (
            doc.page_content
            + str(doc.metadata.get("source", ""))
            + str(doc.metadata.get("page_number", ""))
        )
        return hashlib.md5(payload.encode()).hexdigest()

    def _get_existing_ids(self) -> set:
        """Retrieve all existing document IDs from ChromaDB."""
        try:
            collection = self._client.get_collection(COLLECTION_NAME)
            result = collection.get(include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()
