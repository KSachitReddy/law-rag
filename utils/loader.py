"""
utils/loader.py — PDF document loader for LawAI India.

Loads PDFs using PyPDF and enriches chunks with legal metadata
(act name, chapter, section) extracted heuristically from text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Act name mapping from filename stem ───────────────────────────────────────
FILENAME_TO_ACT: Dict[str, str] = {
    "constitution": "Constitution of India",
    "bns": "Bharatiya Nyaya Sanhita (BNS) 2023",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023",
    "bsa": "Bharatiya Sakshya Adhiniyam (BSA) 2023",
    "it_act": "Information Technology Act 2000",
    "it": "Information Technology Act 2000",
    "companies_act": "Companies Act 2013",
    "companies": "Companies Act 2013",
    "income_tax": "Income Tax Act 1961",
    "income_tax_act": "Income Tax Act 1961",
    "gst": "Goods and Services Tax Act 2017",
    "consumer_protection": "Consumer Protection Act 2019",
    "consumer": "Consumer Protection Act 2019",
    "labour": "Labour Laws",
    "environment": "Environment Protection Act 1986",
    "environmental": "Environment Protection Act 1986",
    "banking": "Banking Regulation Act 1949",
    "cyber": "Cyber Laws",
    "ipr": "Intellectual Property Rights Laws",
    "contract": "Indian Contract Act 1872",
    "family": "Family Law",
    "property": "Transfer of Property Act 1882",
    "criminal": "Criminal Law",
    "civil": "Civil Procedure Code 1908",
    "cpc": "Civil Procedure Code 1908",
    "crpc": "Code of Criminal Procedure",
    "ipc": "Indian Penal Code 1860",
    "evidence": "Indian Evidence Act 1872",
}


class DocumentLoader:
    """
    Loads PDF files, splits them into chunks, and attaches
    rich legal metadata to each chunk.

    Parameters
    ----------
    chunk_size : int
        Target character count per chunk.
    chunk_overlap : int
        Character overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_pdf(
        self,
        file_path: str,
        act_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Load a single PDF and return a list of enriched Document chunks.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the PDF.
        act_name : str, optional
            Override the act name; inferred from filename if omitted.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info("Loading PDF: %s", path.name)

        # Infer act name
        resolved_act = act_name or self._infer_act_name(path.stem)

        try:
            loader = PyPDFLoader(str(path))
            pages: List[Document] = loader.load()
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)
            raise

        logger.info("Loaded %d pages from %s", len(pages), path.name)

        # Split pages into chunks
        chunks: List[Document] = []
        for page in pages:
            page_num = page.metadata.get("page", 0) + 1  # 1-indexed
            splits = self.splitter.split_documents([page])
            for split in splits:
                meta = self._build_metadata(
                    text=split.page_content,
                    source=str(path),
                    page_number=page_num,
                    act_name=resolved_act,
                    original_meta=split.metadata,
                )
                chunks.append(Document(page_content=split.page_content, metadata=meta))

        logger.info("Created %d chunks from %s", len(chunks), path.name)
        return chunks

    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load all PDFs in a directory (optionally recursive).

        Returns a flat list of Document chunks from all files.
        """
        dir_path = Path(directory)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(dir_path.glob(pattern))

        if not pdf_files:
            logger.warning("No PDFs found in %s", directory)
            return []

        all_chunks: List[Document] = []
        for pdf in pdf_files:
            try:
                chunks = self.load_pdf(str(pdf))
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error("Skipping %s due to error: %s", pdf.name, exc)

        logger.info(
            "Total chunks loaded from directory: %d (from %d files)",
            len(all_chunks),
            len(pdf_files),
        )
        return all_chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_act_name(self, stem: str) -> str:
        """Map a filename stem to a human-readable Act name."""
        key = stem.lower().replace("-", "_").replace(" ", "_")
        return FILENAME_TO_ACT.get(key, stem.replace("_", " ").title())

    def _build_metadata(
        self,
        text: str,
        source: str,
        page_number: int,
        act_name: str,
        original_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a rich metadata dict for a chunk.

        Extracts chapter and section numbers heuristically via regex.
        """
        chapter = self._extract_chapter(text)
        section = self._extract_section(text)

        return {
            "source": source,
            "page_number": page_number,
            "act_name": act_name,
            "chapter": chapter,
            "section": section,
            **{k: v for k, v in original_meta.items() if k not in ("page", "source")},
        }

    @staticmethod
    def _extract_chapter(text: str) -> str:
        """Extract chapter number/name from chunk text."""
        patterns = [
            r"CHAPTER\s+([IVXLCDM]+|[0-9]+)",
            r"Chapter\s+([IVXLCDM]+|[0-9]+)",
            r"PART\s+([IVXLCDM]+|[0-9]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text[:300])
            if m:
                return m.group(1)
        return ""

    @staticmethod
    def _extract_section(text: str) -> str:
        """Extract section number from chunk text."""
        patterns = [
            r"Section\s+(\d+[A-Z]?)",
            r"Sec\.\s+(\d+[A-Z]?)",
            r"\bS\.\s*(\d+[A-Z]?)\b",
            r"Article\s+(\d+[A-Z]?)",
            r"^\s*(\d+)\.",          # numbered paragraph at start
        ]
        for pat in patterns:
            m = re.search(pat, text[:400], re.MULTILINE)
            if m:
                return m.group(1)
        return ""
