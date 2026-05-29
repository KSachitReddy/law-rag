"""
utils/retriever.py — Semantic retrieval for LawAI India.

Uses ChromaDB similarity_search_with_score() to find the top-K
most relevant legal chunks for a user query.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from utils.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


class LegalRetriever:
    """
    Retrieves legally relevant document chunks using semantic similarity.

    Parameters
    ----------
    embedding_manager : EmbeddingManager
        Initialised embedding manager with a populated vectorstore.
    top_k : int
        Maximum number of chunks to retrieve per query.
    similarity_threshold : float
        Maximum distance score to accept (lower = more similar in L2/cosine).
        Chunks with score above this threshold are filtered out.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        top_k: int = 5,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.vectorstore = embedding_manager.get_vectorstore()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return enriched result dicts.

        Parameters
        ----------
        query : str
            User question (any supported language).

        Returns
        -------
        list[dict]
            Each dict contains:
            - text        : chunk text
            - score       : similarity distance (lower = better)
            - metadata    : full metadata dict (act_name, page_number, …)
        """
        if not query.strip():
            return []

        logger.info("Retrieving top-%d chunks for query: %s", self.top_k, query[:80])

        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=self.top_k * 2,  # fetch extra to allow filtering
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise

        # Relevance filtering
        filtered = [
            (doc, score)
            for doc, score in results
            if score <= self.similarity_threshold
        ]

        # Sort ascending by score (most similar first)
        filtered.sort(key=lambda x: x[1])

        # Limit to top_k
        filtered = filtered[: self.top_k]

        if not filtered:
            logger.warning("No chunks passed the relevance threshold (%.2f).",
                           self.similarity_threshold)
            # Fall back to raw top results without threshold
            filtered = sorted(results, key=lambda x: x[1])[: self.top_k]

        enriched: List[Dict[str, Any]] = []
        for doc, score in filtered:
            enriched.append(
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "metadata": {
                        "act_name": doc.metadata.get("act_name", "Unknown Act"),
                        "source": doc.metadata.get("source", ""),
                        "page_number": doc.metadata.get("page_number", "?"),
                        "chapter": doc.metadata.get("chapter", ""),
                        "section": doc.metadata.get("section", ""),
                    },
                }
            )
            logger.debug(
                "  [%.4f] %s p.%s — %s",
                score,
                doc.metadata.get("act_name", "?"),
                doc.metadata.get("page_number", "?"),
                doc.page_content[:60],
            )

        logger.info("Retrieved %d relevant chunks.", len(enriched))
        return enriched
