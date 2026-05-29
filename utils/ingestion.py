"""
utils/ingestion.py — Document ingestion pipeline for LawAI India.

Orchestrates loading → chunking → embedding → ChromaDB storage.
Supports single-file and batch ingestion, plus folder monitoring
for the /data/new_laws/ auto-update system.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from utils.loader import DocumentLoader
from utils.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full ingestion lifecycle for legal PDFs.

    Parameters
    ----------
    embedder : EmbeddingManager
        Initialised embedding manager.
    data_dir : str
        Root data directory containing PDF files.
    chunk_size : int
        Character count per text chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.
    """

    def __init__(
        self,
        embedder: EmbeddingManager,
        data_dir: str = "data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.embedder = embedder
        self.data_dir = Path(data_dir)
        self.loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_all(self) -> Dict[str, Any]:
        """
        Ingest every PDF found recursively under ``data_dir``.

        Returns
        -------
        dict
            ``total_files``, ``total_chunks``, ``skipped``, ``errors``
        """
        pdf_files = sorted(self.data_dir.rglob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found under %s", self.data_dir)
            return {"total_files": 0, "total_chunks": 0, "skipped": 0, "errors": 0}

        logger.info("Starting batch ingestion of %d PDFs…", len(pdf_files))
        total_chunks = 0
        errors = 0
        skipped = 0

        for pdf in pdf_files:
            result = self.ingest_single(str(pdf))
            added = result.get("chunks_added", 0)
            if added == 0 and result.get("status") == "skipped":
                skipped += 1
            elif result.get("status") == "error":
                errors += 1
            else:
                total_chunks += added

        summary = {
            "total_files": len(pdf_files),
            "total_chunks": total_chunks,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info("Batch ingestion complete: %s", summary)
        return summary

    def ingest_single(
        self,
        file_path: str,
        act_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single PDF file.

        Parameters
        ----------
        file_path : str
            Path to the PDF.
        act_name : str, optional
            Human-readable act name (inferred from filename if omitted).

        Returns
        -------
        dict
            ``file``, ``chunks_added``, ``status``
        """
        path = Path(file_path)
        logger.info("Ingesting: %s", path.name)

        try:
            t0 = time.perf_counter()
            documents = self.loader.load_pdf(file_path, act_name=act_name)
            chunks_added = self.embedder.add_documents(documents)
            elapsed = time.perf_counter() - t0

            if chunks_added == 0:
                status = "skipped"
                logger.info("All chunks already indexed for %s.", path.name)
            else:
                status = "ok"
                logger.info(
                    "Ingested %d chunks from %s in %.1fs.", chunks_added, path.name, elapsed
                )

            return {
                "file": path.name,
                "chunks_added": chunks_added,
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
            }

        except FileNotFoundError as exc:
            logger.error("File not found: %s", exc)
            return {"file": path.name, "chunks_added": 0, "status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Unexpected error ingesting %s: %s", path.name, exc)
            return {"file": path.name, "chunks_added": 0, "status": "error", "error": str(exc)}

    def monitor_new_laws(self, interval_seconds: int = 60) -> None:
        """
        Continuously monitor ``data/new_laws/`` and auto-ingest new PDFs.

        This is intended to be run in a background thread or separate process.

        Parameters
        ----------
        interval_seconds : int
            Poll interval in seconds.
        """
        new_laws_dir = self.data_dir / "new_laws"
        new_laws_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Monitoring %s for new PDFs (poll every %ds)…",
            new_laws_dir,
            interval_seconds,
        )

        seen: set = set()

        while True:
            try:
                pdfs: List[Path] = sorted(new_laws_dir.glob("*.pdf"))
                for pdf in pdfs:
                    if pdf.name not in seen:
                        logger.info("New PDF detected: %s", pdf.name)
                        result = self.ingest_single(str(pdf))
                        if result["status"] in ("ok", "skipped"):
                            seen.add(pdf.name)
                        logger.info("Auto-ingestion result: %s", result)
            except Exception as exc:
                logger.error("Monitor error: %s", exc)

            time.sleep(interval_seconds)
